#!/usr/bin/env python3
"""
go1_log_analyser.py — Comprehensive Go1 policy log visualiser
Works with both sim (.npz from play.py --log) and real hardware logs.

Usage:
    python go1_log_analyser.py --file_path <log.npz>
    python go1_log_analyser.py --file_path <log.npz> --out_dir ./plots

Outputs:
    Page 1: Overview summary (velocity, tilt, contact, reward)
    Page 2: RL_th and FR_th detailed analysis (stiction / binding)
    Page 3: All 12 joints — tracking error and delta patterns
    Page 4: Gait analysis (foot contact timing, air time, frequency)
    Page 5: Rate weight evidence (|Δdelta| per joint comparison)
    Page 6: Observation space channels (raw_net saturation, key obs)
    Page 7: Lateral drift and body stability
    summary.txt: all key metrics printed to file
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

# ── Joint / axis names ─────────────────────────────────────────────────────
JOINTS = [
    "FL_hip", "FR_hip", "RL_hip", "RR_hip",
    "FL_th",  "FR_th",  "RL_th",  "RR_th",
    "FL_kn",  "FR_kn",  "RL_kn",  "RR_kn",
]
FEET   = ["FL", "FR", "RL", "RR"]
JSHORT = ["FLh","FRh","RLh","RRh","FLt","FRt","RLt","RRt","FLk","FRk","RLk","RRk"]

# ── Colour scheme ──────────────────────────────────────────────────────────
C = {
    "FL":  "#1565C0",   # blue
    "FR":  "#2E7D32",   # green
    "RL":  "#C62828",   # red   ← fault joint
    "RR":  "#F57F17",   # amber
    "cmd": "#555555",
    "tgt": "#90CAF9",
    "act": "#1565C0",
    "tilt":"#C62828",
    "good":"#2E7D32",
    "bad": "#C62828",
    "warn":"#F57F17",
    "shad":"#BBDEFB",
}
JCOL = [C["FL"],C["FR"],C["RL"],C["RR"]] * 3   # hip/thigh/knee inherit foot colour


# ═══════════════════════════════════════════════════════════════════════════
def load(path):
    d   = np.load(path, allow_pickle=True)
    out = {k: d[k] for k in d.keys()}
    N   = out["target_q"].shape[0]
    dt  = float(out["step_dt"].item()) if "step_dt" in out else 0.02
    t   = np.arange(N) * dt

    # Derived
    out["N"]    = N
    out["dt"]   = dt
    out["t"]    = t
    out["dur"]  = t[-1]
    out["src"]  = str(out["src"][0]) if "src" in out else "unknown"
    out["is_real"] = (out["src"] == "real")

    dq = out["default_q"] if "default_q" in out else np.array(
        [0.1,0.1,0.1,0.1, 0.8,0.8,0.8,0.8, -1.5,-1.5,-1.5,-1.5], np.float32)
    out["default_q"] = dq

    out["track_err"] = np.abs(out["actual_q"] - out["target_q"])
    out["ddelta"]    = np.abs(np.diff(out["tanh_delta"], axis=0, prepend=out["tanh_delta"][[0]]))
    out["feet_bin"]  = (out["contact"] > 1.0).astype(float)
    out["n_feet"]    = out["feet_bin"].sum(axis=1)

    # Tilt
    if "tilt_deg" not in out:
        g = out["proj_grav"]
        out["tilt_deg"] = np.degrees(np.sqrt(g[:,0]**2 + g[:,1]**2))

    # Stall events per joint (commanded but not moving)
    stall = np.zeros((N, 12), bool)
    for j in range(12):
        stall[:, j] = (
            (np.abs(out["actual_qd"][:,j]) < 0.05) &
            (np.abs(out["tanh_delta"][:,j]) > 0.05)
        )
    out["stall"] = stall

    # Lin vel (sim only)
    out["has_linvel"] = "lin_vel" in out and out["lin_vel"].shape[1] >= 1

    return out


# ═══════════════════════════════════════════════════════════════════════════
def summary_text(d):
    lines = []
    src  = "REAL HARDWARE" if d["is_real"] else "SIMULATION"
    lines.append(f"{'='*60}")
    lines.append(f"Go1 Policy Log Analysis — {src}")
    lines.append(f"Duration:   {d['dur']:.1f}s  ({d['N']} steps @ {1/d['dt']:.0f}Hz)")
    lines.append(f"{'='*60}")

    lines.append("\n--- VELOCITY ---")
    if d["has_linvel"]:
        lv = d["lin_vel"][:,0]
        cv = d["cmd"][:,0]
        lines.append(f"cmd_vx:     mean={cv.mean():.3f}  range=[{cv.min():.3f},{cv.max():.3f}]")
        lines.append(f"act_vx:     mean={lv.mean():.3f}  std={lv.std():.3f}")
        lines.append(f"track_err:  {np.abs(lv-cv).mean():.3f} m/s")
    else:
        lines.append(f"cmd_vx:     mean={d['cmd'][:,0].mean():.3f}")

    lines.append("\n--- STABILITY ---")
    tilt = d["tilt_deg"]
    lines.append(f"Tilt mean:  {tilt.mean():.2f}°  max={tilt.max():.2f}°  std={tilt.std():.2f}°")
    lines.append(f">10°:       {(tilt>10).mean()*100:.1f}%")
    lines.append(f">20°:       {(tilt>20).mean()*100:.1f}%")

    lines.append("\n--- GAIT ---")
    nf = d["n_feet"]
    lines.append(f"Avg feet on ground: {nf.mean():.2f}")
    lines.append(f"2-foot trot:        {(nf==2).mean()*100:.1f}%")
    lines.append(f"3-foot:             {(nf==3).mean()*100:.1f}%")
    lines.append(f"4-foot (stand):     {(nf==4).mean()*100:.1f}%")
    lines.append(f"0-foot (air):       {(nf==0).mean()*100:.1f}%")

    lines.append("\n--- STICTION (RL_th) ---")
    st_rl = d["stall"][:,6]
    lines.append(f"RL_th stall:        {st_rl.mean()*100:.1f}%  ({st_rl.sum()}/{d['N']} steps)")
    lines.append(f"RL_th |Ddelta|:     {d['ddelta'][:,6].mean():.4f}")
    lines.append(f"RL_th track_err:    mean={d['track_err'][:,6].mean():.4f}  max={d['track_err'][:,6].max():.4f}")
    lines.append(f"RL_th vel_abs:      {np.abs(d['actual_qd'][:,6]).mean():.4f} rad/s")
    lines.append(f"FL_th stall (ref):  {d['stall'][:,4].mean()*100:.1f}%")
    lines.append(f"FL_th |Ddelta|:     {d['ddelta'][:,4].mean():.4f}")

    lines.append("\n--- FR_th BINDING ---")
    fr = d["actual_q"][:,5]
    lines.append(f"FR_th actual max:   {fr.max():.4f} rad")
    lines.append(f"FR_th >0.80:        {(fr>0.80).mean()*100:.1f}%")
    lines.append(f"FR_th >0.82:        {(fr>0.82).mean()*100:.1f}%")

    lines.append("\n--- LATERAL DRIFT ---")
    gx = d["proj_grav"][:,0]
    gy = d["proj_grav"][:,1]
    lines.append(f"grav_x mean:        {gx.mean():.4f}  (+=right)")
    lines.append(f"grav_y mean:        {gy.mean():.4f}  (+=forward)")
    lines.append(f"Lean right >0.05:   {(gx>0.05).mean()*100:.1f}%")
    lines.append(f"Lean left  <-0.05:  {(gx<-0.05).mean()*100:.1f}%")

    lines.append("\n--- ALL JOINTS (stall / |Ddelta| / track_err) ---")
    for i, jn in enumerate(JOINTS):
        st  = d["stall"][:,i].mean()*100
        dde = d["ddelta"][:,i].mean()
        te  = d["track_err"][:,i].mean()
        marker = " ← FAULT" if i==6 else (" ← BINDING" if i==5 else "")
        lines.append(f"  {jn:<8}: stall={st:5.1f}%  |Dd|={dde:.4f}  te={te:.4f}{marker}")

    if "raw_net" in d:
        rn = d["raw_net"]
        lines.append("\n--- RAW NET SATURATION (|rn|>2.5) ---")
        for i, jn in enumerate(JOINTS):
            sat = (np.abs(rn[:,i])>2.5).mean()*100
            if sat > 5:
                lines.append(f"  {jn:<8}: {sat:.1f}%  *** high")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
def page1_overview(d, fig_path):
    fig = plt.figure(figsize=(14, 10), dpi=150)
    fig.suptitle(f"Page 1 — Overview  [{d['src'].upper()} | {d['dur']:.1f}s]",
                 fontsize=11, fontweight="bold")
    gs = gridspec.GridSpec(3, 3, figure=fig,
                           hspace=0.45, wspace=0.35,
                           left=0.07, right=0.97, top=0.92, bottom=0.07)
    t  = d["t"]

    # 1. Velocity
    ax = fig.add_subplot(gs[0, :2])
    if d["has_linvel"]:
        ax.plot(t, d["cmd"][:,0], "--", color=C["cmd"], lw=1.2, label="cmd vx")
        ax.plot(t, d["lin_vel"][:,0], color=C["FL"], lw=1.5, label="act vx")
        ax.fill_between(t, d["cmd"][:,0], d["lin_vel"][:,0],
                        alpha=0.15, color=C["RL"], label="error")
    else:
        ax.plot(t, d["cmd"][:,0], "--", color=C["cmd"], lw=1.2, label="cmd vx")
        ax.text(0.5, 0.5, "lin_vel not available (real log)",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=9, color="grey")
    ax.set_ylabel("m/s", fontsize=8)
    ax.set_title("Forward Velocity", fontsize=9, fontweight="bold")
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)

    # 2. Tilt
    ax = fig.add_subplot(gs[0, 2])
    tilt = d["tilt_deg"]
    ax.plot(t, tilt, color=C["tilt"], lw=1.2)
    ax.axhline(10, color=C["warn"], ls="--", lw=0.8, label="10°")
    ax.axhline(20, color=C["bad"],  ls="--", lw=0.8, label="20°")
    ax.fill_between(t, 0, tilt, alpha=0.2, color=C["tilt"])
    ax.set_ylabel("degrees", fontsize=8)
    ax.set_title(f"Body Tilt (mean={tilt.mean():.1f}° max={tilt.max():.1f}°)",
                 fontsize=9, fontweight="bold")
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)

    # 3. Contact heatmap
    ax = fig.add_subplot(gs[1, :2])
    im = ax.imshow(d["feet_bin"].T, aspect="auto", cmap="Blues",
                   extent=[t[0], t[-1], -0.5, 3.5], vmin=0, vmax=1)
    ax.set_yticks([0,1,2,3])
    ax.set_yticklabels(FEET[::-1], fontsize=8)
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_title("Foot Contact (blue=on ground)", fontsize=9, fontweight="bold")
    ax.tick_params(labelsize=7)

    # 4. N feet on ground
    ax = fig.add_subplot(gs[1, 2])
    nf = d["n_feet"]
    for v, col, lab in [(2,C["good"],"trot(2)"),(3,C["warn"],"3-foot"),(4,C["bad"],"stand(4)")]:
        ax.fill_between(t, 0, (nf==v).astype(float)*v,
                        alpha=0.5, color=col, label=lab)
    ax.plot(t, nf, color="k", lw=0.7, alpha=0.5)
    ax.set_ylabel("feet", fontsize=8)
    ax.set_ylim(-0.1, 5)
    ax.set_title(f"Feet on Ground (trot={( nf==2).mean()*100:.0f}%)",
                 fontsize=9, fontweight="bold")
    ax.legend(fontsize=7, loc="upper right")
    ax.tick_params(labelsize=7)

    # 5. Projected gravity (lateral + forward lean)
    ax = fig.add_subplot(gs[2, :2])
    ax.plot(t, d["proj_grav"][:,0], color=C["RL"],  lw=1.2, label="grav_x (roll)")
    ax.plot(t, d["proj_grav"][:,1], color=C["FL"],  lw=1.2, label="grav_y (pitch)")
    ax.axhline(0, color="k", lw=0.5)
    ax.axhline( 0.05, color=C["warn"], ls=":", lw=0.8)
    ax.axhline(-0.05, color=C["warn"], ls=":", lw=0.8)
    ax.set_ylabel("proj_grav", fontsize=8)
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_title("Body Lean (0=upright)", fontsize=9, fontweight="bold")
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)

    # 6. Reward (sim only)
    ax = fig.add_subplot(gs[2, 2])
    if "reward" in d and d["reward"].sum() != 0:
        ax.plot(t, d["reward"], color=C["FL"], lw=1.0)
        ax.set_ylabel("reward/step", fontsize=8)
        ax.set_title(f"Reward (mean={d['reward'].mean():.2f})",
                     fontsize=9, fontweight="bold")
    else:
        ax.text(0.5, 0.5, "reward not in log", transform=ax.transAxes,
                ha="center", va="center", fontsize=9, color="grey")
        ax.set_title("Reward", fontsize=9, fontweight="bold")
    ax.tick_params(labelsize=7)

    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fig_path}")


# ═══════════════════════════════════════════════════════════════════════════
def page2_fault_joints(d, fig_path):
    fig = plt.figure(figsize=(14, 12), dpi=150)
    fig.suptitle("Page 2 — Fault Joint Analysis: RL_th (stiction) + FR_th (binding)",
                 fontsize=11, fontweight="bold")
    gs = gridspec.GridSpec(4, 2, figure=fig,
                           hspace=0.50, wspace=0.35,
                           left=0.07, right=0.97, top=0.93, bottom=0.07)
    t = d["t"]

    for col, jidx, jname, cap, cap_label in [
        (0, 6, "RL_th", None, "stiction"),
        (1, 5, "FR_th", 0.82, "binding"),
    ]:
        dq = d["default_q"][jidx]
        tgt = d["target_q"][:,jidx]
        act = d["actual_q"][:,jidx]
        vel = d["actual_qd"][:,jidx]
        dde = d["ddelta"][:,jidx]
        stall = d["stall"][:,jidx]

        # Row 0: target vs actual
        ax = fig.add_subplot(gs[0, col])
        ax.plot(t, tgt, color=C["tgt"], lw=1.0, label="target", alpha=0.8)
        ax.plot(t, act, color=JCOL[jidx], lw=1.5, label="actual")
        if cap is not None:
            ax.axhline(cap, color=C["bad"], ls="--", lw=1.0, label=f"{cap_label} {cap}")
        ax.axhline(dq, color=C["cmd"], ls=":", lw=0.8, label=f"default {dq}")
        ax.fill_between(t, tgt, act, alpha=0.2, color=C["RL"])
        ax.set_ylabel("rad", fontsize=8)
        ax.set_title(f"{jname} — Target vs Actual", fontsize=9, fontweight="bold")
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)

        # Row 1: tracking error + stall events
        ax = fig.add_subplot(gs[1, col])
        te = np.abs(tgt - act)
        ax.plot(t, te, color=JCOL[jidx], lw=1.2, label="track_err")
        stall_t = t[stall]
        if len(stall_t):
            ax.scatter(stall_t, te[stall], color=C["bad"], s=8,
                       zorder=5, label=f"stall ({stall.mean()*100:.1f}%)")
        ax.set_ylabel("rad", fontsize=8)
        ax.set_title(f"{jname} — Tracking Error (stall={stall.mean()*100:.1f}%)",
                     fontsize=9, fontweight="bold")
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)

        # Row 2: velocity
        ax = fig.add_subplot(gs[2, col])
        ax.plot(t, vel, color=JCOL[jidx], lw=1.0)
        ax.axhline(0, color="k", lw=0.5)
        ax.fill_between(t, 0, vel, alpha=0.25, color=JCOL[jidx])
        ax.set_ylabel("rad/s", fontsize=8)
        ax.set_title(f"{jname} — Joint Velocity (mean_abs={np.abs(vel).mean():.3f})",
                     fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7)

        # Row 3: delta pattern (stiction-breaking impulses)
        ax = fig.add_subplot(gs[3, col])
        ax.plot(t, d["tanh_delta"][:,jidx], color=JCOL[jidx], lw=1.0, label="delta")
        ax.fill_between(t, 0, d["tanh_delta"][:,jidx], alpha=0.2, color=JCOL[jidx])
        ax2 = ax.twinx()
        ax2.plot(t, dde, color=C["bad"], lw=0.8, alpha=0.6, label="|Δdelta|")
        ax2.set_ylabel("|Δdelta|", fontsize=7, color=C["bad"])
        ax2.tick_params(labelsize=6, colors=C["bad"])
        ax.set_ylabel("delta (rad)", fontsize=8)
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_title(f"{jname} — Action Pattern (|Δd|={dde.mean():.4f})",
                     fontsize=9, fontweight="bold")
        ax.legend(fontsize=7, loc="upper left")
        ax.tick_params(labelsize=7)

    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fig_path}")


# ═══════════════════════════════════════════════════════════════════════════
def page3_all_joints(d, fig_path):
    fig, axes = plt.subplots(4, 3, figsize=(15, 14), dpi=150)
    fig.suptitle("Page 3 — All 12 Joints: Target vs Actual + Tracking Error",
                 fontsize=11, fontweight="bold")
    t = d["t"]
    for idx, (ax, jn) in enumerate(zip(axes.flat, JOINTS)):
        tgt = d["target_q"][:,idx]
        act = d["actual_q"][:,idx]
        te  = np.abs(tgt - act)
        ax.plot(t, tgt, color="#90CAF9", lw=0.9, label="tgt", alpha=0.8)
        ax.plot(t, act, color=JCOL[idx], lw=1.3, label="act")
        ax2 = ax.twinx()
        ax2.fill_between(t, 0, te, alpha=0.25, color=C["RL"])
        ax2.set_ylabel("err", fontsize=6, color=C["RL"])
        ax2.tick_params(labelsize=5, colors=C["RL"])
        stall = d["stall"][:,idx]
        if stall.any():
            ax.scatter(t[stall], act[stall], color=C["bad"],
                       s=5, zorder=5, alpha=0.5)
        fault_label = " ★FAULT" if idx==6 else (" ★BIND" if idx==5 else "")
        ax.set_title(f"{jn}{fault_label}  te={te.mean():.3f}  stall={stall.mean()*100:.0f}%",
                     fontsize=8, fontweight="bold" if idx in [5,6] else "normal")
        ax.legend(fontsize=6, loc="upper right")
        ax.tick_params(labelsize=6)
        ax.set_ylabel("rad", fontsize=7)
        if idx >= 9:
            ax.set_xlabel("t (s)", fontsize=7)
    plt.tight_layout(rect=[0,0,1,0.96])
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fig_path}")


# ═══════════════════════════════════════════════════════════════════════════
def page4_gait(d, fig_path):
    fig = plt.figure(figsize=(14, 10), dpi=150)
    fig.suptitle("Page 4 — Gait Analysis", fontsize=11, fontweight="bold")
    gs = gridspec.GridSpec(3, 2, figure=fig,
                           hspace=0.45, wspace=0.35,
                           left=0.07, right=0.97, top=0.92, bottom=0.07)
    t   = d["t"]
    fb  = d["feet_bin"]
    FCOLS = [C["FL"],C["FR"],C["RL"],C["RR"]]

    # Contact force per foot
    ax = fig.add_subplot(gs[0, :])
    for i, (fn, fc) in enumerate(zip(FEET, FCOLS)):
        offset = i * 2.5
        ax.plot(t, d["contact"][:,i] / d["contact"].max() * 2 + offset,
                color=fc, lw=1.0, label=fn)
        ax.axhline(offset, color="grey", lw=0.3)
    ax.set_yticks([i*2.5 + 1.0 for i in range(4)])
    ax.set_yticklabels(FEET, fontsize=8)
    ax.set_title("Foot Contact Forces (normalised)", fontsize=9, fontweight="bold")
    ax.tick_params(labelsize=7)

    # Phase diagram: diagonal pairs
    ax = fig.add_subplot(gs[1, 0])
    diag1 = fb[:,0] * fb[:,3]   # FL + RR (diagonal A)
    diag2 = fb[:,1] * fb[:,2]   # FR + RL (diagonal B)
    ax.fill_between(t, 0, diag1, alpha=0.5, color=C["FL"], label="FL+RR diag")
    ax.fill_between(t, 0, diag2, alpha=0.5, color=C["FR"], label="FR+RL diag")
    ax.set_ylabel("contact", fontsize=8)
    ax.set_title(f"Trot Diagonals (FL+RR={diag1.mean()*100:.0f}%  FR+RL={diag2.mean()*100:.0f}%)",
                 fontsize=9, fontweight="bold")
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)

    # N feet histogram
    ax = fig.add_subplot(gs[1, 1])
    nf = d["n_feet"]
    vals, counts = np.unique(nf.astype(int), return_counts=True)
    pcts  = counts / len(nf) * 100
    bars  = ax.bar(vals, pcts, color=[C["bad"],C["warn"],C["good"],C["FL"],C["RL"]][:len(vals)])
    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{pct:.0f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Feet on ground", fontsize=8)
    ax.set_ylabel("%", fontsize=8)
    ax.set_title("Contact Distribution", fontsize=9, fontweight="bold")
    ax.tick_params(labelsize=7)

    # Angular velocity (roll/pitch/yaw)
    ax = fig.add_subplot(gs[2, :])
    labels = ["roll(x)", "pitch(y)", "yaw(z)"]
    cols   = [C["RL"], C["FL"], C["FR"]]
    for i, (lab, col) in enumerate(zip(labels, cols)):
        ax.plot(t, d["ang_vel"][:,i], color=col, lw=1.0, label=lab, alpha=0.8)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("rad/s", fontsize=8)
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_title("Body Angular Velocity", fontsize=9, fontweight="bold")
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)

    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fig_path}")


# ═══════════════════════════════════════════════════════════════════════════
def page5_rate_weights(d, fig_path):
    fig = plt.figure(figsize=(14, 10), dpi=150)
    fig.suptitle("Page 5 — Rate Weight Evidence: |Δdelta| per Joint",
                 fontsize=11, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig,
                           hspace=0.40, wspace=0.35,
                           left=0.07, right=0.97, top=0.92, bottom=0.07)
    t   = d["t"]
    dde = d["ddelta"]

    # Bar chart: mean |Ddelta| per joint
    ax = fig.add_subplot(gs[0, 0])
    means = dde.mean(axis=0)
    bars  = ax.bar(range(12), means, color=JCOL)
    ax.bar(6, means[6], color=C["bad"],  label="RL_th (fault)")
    ax.bar(5, means[5], color=C["warn"], label="FR_th (binding)")
    ax.set_xticks(range(12))
    ax.set_xticklabels(JSHORT, fontsize=7, rotation=45)
    ax.set_ylabel("mean |Δdelta|", fontsize=8)
    ax.set_title("Mean Action Change per Joint\n(higher RL_th = impulsive stiction-breaking)",
                 fontsize=9, fontweight="bold")
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)

    # Time series |Ddelta| for thigh joints
    ax = fig.add_subplot(gs[0, 1])
    thigh_cols = [C["FL"],C["FR"],C["RL"],C["RR"]]
    thigh_labels = ["FL_th","FR_th","RL_th★","RR_th"]
    for i, (col, lab) in enumerate(zip(thigh_cols, thigh_labels)):
        jidx = i + 4
        ax.plot(t, dde[:,jidx], color=col, lw=0.8, alpha=0.8, label=lab)
    ax.set_ylabel("|Δdelta| rad/step", fontsize=8)
    ax.set_title("Thigh |Δdelta| over Time\n(RL_th★ should be highest in calibrated policy)",
                 fontsize=9, fontweight="bold")
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)

    # RL_th vs FL_th impulse comparison
    ax = fig.add_subplot(gs[1, 0])
    bins = np.linspace(0, max(dde[:,4].max(), dde[:,6].max()) * 1.05, 40)
    ax.hist(dde[:,4], bins=bins, color=C["FL"], alpha=0.6, label=f"FL_th (healthy) mean={dde[:,4].mean():.4f}")
    ax.hist(dde[:,6], bins=bins, color=C["RL"], alpha=0.6, label=f"RL_th (fault) mean={dde[:,6].mean():.4f}")
    ax.axvline(0.05, color="k", ls="--", lw=0.8, label="impulse threshold 0.05")
    ax.set_xlabel("|Δdelta| rad/step", fontsize=8)
    ax.set_ylabel("count", fontsize=8)
    ax.set_title("RL_th vs FL_th Action Change Distribution",
                 fontsize=9, fontweight="bold")
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)

    # Stall rate bar chart
    ax = fig.add_subplot(gs[1, 1])
    stall_rates = d["stall"].mean(axis=0) * 100
    bars = ax.bar(range(12), stall_rates, color=JCOL)
    ax.bar(6, stall_rates[6], color=C["bad"],  label=f"RL_th {stall_rates[6]:.1f}%")
    ax.bar(5, stall_rates[5], color=C["warn"], label=f"FR_th {stall_rates[5]:.1f}%")
    ax.set_xticks(range(12))
    ax.set_xticklabels(JSHORT, fontsize=7, rotation=45)
    ax.set_ylabel("stall rate %", fontsize=8)
    ax.set_title("Stall Rate per Joint\n(commanded but not moving)",
                 fontsize=9, fontweight="bold")
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)

    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fig_path}")


# ═══════════════════════════════════════════════════════════════════════════
def page6_network(d, fig_path):
    fig = plt.figure(figsize=(14, 10), dpi=150)
    fig.suptitle("Page 6 — Network Output (raw_net saturation + obs channels)",
                 fontsize=11, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig,
                           hspace=0.40, wspace=0.35,
                           left=0.07, right=0.97, top=0.92, bottom=0.07)
    t = d["t"]

    if "raw_net" in d:
        rn = d["raw_net"]

        # Saturation heatmap
        ax = fig.add_subplot(gs[0, 0])
        sat = (np.abs(rn) > 2.5).astype(float)
        im  = ax.imshow(sat.T, aspect="auto", cmap="Reds",
                        extent=[t[0], t[-1], -0.5, 11.5], vmin=0, vmax=1)
        ax.set_yticks(range(12))
        ax.set_yticklabels(JSHORT[::-1], fontsize=7)
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_title("Raw Net Saturation (|rn|>2.5 = red)",
                     fontsize=9, fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.03)
        ax.tick_params(labelsize=7)

        # Saturation rate bar
        ax = fig.add_subplot(gs[0, 1])
        sat_rates = (np.abs(rn) > 2.5).mean(axis=0) * 100
        bars = ax.bar(range(12), sat_rates, color=JCOL)
        ax.bar(6, sat_rates[6], color=C["bad"])
        ax.set_xticks(range(12))
        ax.set_xticklabels(JSHORT, fontsize=7, rotation=45)
        ax.set_ylabel("% saturated", fontsize=8)
        ax.set_title("Network Output Saturation per Joint\n(high = policy trying to command more than tanh allows)",
                     fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7)

        # RL_th raw_net time series
        ax = fig.add_subplot(gs[1, 0])
        ax.plot(t, rn[:,6], color=C["RL"], lw=1.0, label="RL_th raw_net")
        ax.plot(t, rn[:,4], color=C["FL"], lw=0.8, alpha=0.7, label="FL_th raw_net")
        ax.axhline( 2.5, color="k", ls="--", lw=0.7, label="±2.5 sat")
        ax.axhline(-2.5, color="k", ls="--", lw=0.7)
        ax.set_ylabel("raw network output", fontsize=8)
        ax.set_title("RL_th vs FL_th Network Output",
                     fontsize=9, fontweight="bold")
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)
    else:
        for i in range(3):
            ax = fig.add_subplot(gs.new_subplotspec((i//2, i%2)))
            ax.text(0.5,0.5,"raw_net not in log",transform=ax.transAxes,
                    ha="center",va="center",color="grey")

    # Tanh delta heatmap
    ax = fig.add_subplot(gs[1, 1])
    im = ax.imshow(d["tanh_delta"].T, aspect="auto", cmap="RdBu_r",
                   extent=[t[0], t[-1], -0.5, 11.5],
                   vmin=-0.35, vmax=0.35)
    ax.set_yticks(range(12))
    ax.set_yticklabels(JSHORT[::-1], fontsize=7)
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_title("Action (tanh_delta) Heatmap\n(blue=neg, red=pos, white=zero)",
                 fontsize=9, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.03)
    ax.tick_params(labelsize=7)

    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fig_path}")


# ═══════════════════════════════════════════════════════════════════════════
def page7_stability(d, fig_path):
    fig = plt.figure(figsize=(14, 10), dpi=150)
    fig.suptitle("Page 7 — Body Stability and Lateral Drift",
                 fontsize=11, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig,
                           hspace=0.40, wspace=0.35,
                           left=0.07, right=0.97, top=0.92, bottom=0.07)
    t = d["t"]

    # Tilt time series with zones
    ax = fig.add_subplot(gs[0, :])
    tilt = d["tilt_deg"]
    ax.fill_between(t, 0,  10, alpha=0.08, color=C["good"], label="safe <10°")
    ax.fill_between(t, 10, 20, alpha=0.08, color=C["warn"], label="warn 10-20°")
    ax.fill_between(t, 20, max(tilt.max()*1.1, 25),
                    alpha=0.08, color=C["bad"],  label="danger >20°")
    ax.plot(t, tilt, color=C["tilt"], lw=1.5)
    ax.axhline(10, color=C["warn"], ls="--", lw=0.8)
    ax.axhline(20, color=C["bad"],  ls="--", lw=0.8)
    ax.set_ylabel("degrees", fontsize=8)
    ax.set_title(f"Body Tilt — mean={tilt.mean():.1f}°  max={tilt.max():.1f}°"
                 f"  >10°={(tilt>10).mean()*100:.0f}%  >20°={(tilt>20).mean()*100:.0f}%",
                 fontsize=9, fontweight="bold")
    ax.legend(fontsize=7, loc="upper right")
    ax.tick_params(labelsize=7)

    # Tilt histogram
    ax = fig.add_subplot(gs[1, 0])
    ax.hist(tilt, bins=30, color=C["tilt"], alpha=0.7, edgecolor="white")
    ax.axvline(10, color=C["warn"], ls="--", lw=1.0, label="10°")
    ax.axvline(20, color=C["bad"],  ls="--", lw=1.0, label="20°")
    ax.set_xlabel("Tilt (°)", fontsize=8)
    ax.set_ylabel("count", fontsize=8)
    ax.set_title("Tilt Distribution", fontsize=9, fontweight="bold")
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)

    # Phase portrait: grav_x vs grav_y
    ax = fig.add_subplot(gs[1, 1])
    gx = d["proj_grav"][:,0]
    gy = d["proj_grav"][:,1]
    sc = ax.scatter(gx, gy, c=t, cmap="viridis", s=4, alpha=0.6)
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    circle = plt.Circle((0,0), 0.1, color=C["good"], fill=False,
                         ls="--", lw=0.8, label="r=0.1 (stable)")
    ax.add_patch(circle)
    ax.set_aspect("equal")
    ax.set_xlabel("grav_x (roll)", fontsize=8)
    ax.set_ylabel("grav_y (pitch)", fontsize=8)
    ax.set_title("Tilt Phase Portrait (colour=time)\n(good policy: dense cluster near origin)",
                 fontsize=9, fontweight="bold")
    plt.colorbar(sc, ax=ax, label="time", fraction=0.03)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)

    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fig_path}")


# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Go1 policy log analyser — sim and real hardware .npz files")
    parser.add_argument("--file_path", required=True,
                        help="Path to .npz log file from play.py --log or go1_deploy.py")
    parser.add_argument("--out_dir",   default=None,
                        help="Output directory for plots (default: same folder as npz)")
    args = parser.parse_args()

    if not os.path.isfile(args.file_path):
        print(f"[ERROR] File not found: {args.file_path}")
        return

    # Output directory
    base = os.path.splitext(os.path.basename(args.file_path))[0]
    out  = args.out_dir if args.out_dir else os.path.dirname(
        os.path.abspath(args.file_path))
    os.makedirs(out, exist_ok=True)

    print(f"\nLoading: {args.file_path}")
    d = load(args.file_path)
    src = "REAL" if d["is_real"] else "SIM"
    print(f"  Source: {src}  |  Duration: {d['dur']:.1f}s  |  Steps: {d['N']}")

    # Summary text
    txt_path = os.path.join(out, f"{base}_summary.txt")
    summary  = summary_text(d)
    with open(txt_path, "w") as f:
        f.write(summary)
    print(f"\n{'='*60}")
    print(summary)
    print(f"{'='*60}")
    print(f"\n  Saved summary: {txt_path}")

    # All plot pages
    pages = [
        (page1_overview,    "page1_overview.png"),
        (page2_fault_joints,"page2_fault_joints.png"),
        (page3_all_joints,  "page3_all_joints.png"),
        (page4_gait,        "page4_gait.png"),
        (page5_rate_weights,"page5_rate_weights.png"),
        (page6_network,     "page6_network.png"),
        (page7_stability,   "page7_stability.png"),
    ]
    print(f"\nGenerating {len(pages)} plot pages...")
    for fn, fname in pages:
        fp = os.path.join(out, f"{base}_{fname}")
        fn(d, fp)

    print(f"\nDone. All outputs in: {out}/")
    print(f"  Files: {base}_summary.txt + {base}_page1-7.png")


if __name__ == "__main__":
    main()
