#!/usr/bin/env python3
"""
find_best_checkpoint.py — Rank all Go1 checkpoints from TFEvents log.

Reads the TensorBoard events file to extract per-iteration metrics,
scores every saved checkpoint, and prints a ranked deployment shortlist.
No GPU, no Isaac, no eval rollouts needed.

Usage:
    python find_best_checkpoint.py --log_dir <path_to_run_folder>

    # Your run:
    python find_best_checkpoint.py \\
        --log_dir ~/IsaacLab/scripts/reinforcement_learning/rsl_rl/logs/rsl_rl/go1_himloco/2026-04-09_16-09-01

Output:
    1. Per-metric plots saved as  best_checkpoint_analysis.png
    2. Top-10 candidate table printed to stdout
    3. best_checkpoint_candidates.txt  written to log_dir
"""

import os
import sys
import glob
import argparse
import numpy as np

# ── Argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument(
    "--log_dir", required=True,
    help="Path to the training run folder containing events.out.tfevents.*")
parser.add_argument(
    "--out_dir", default=None,
    help="Where to save plots/report (default: same as log_dir)")
parser.add_argument(
    "--iters_per_ckpt", type=int, default=100,
    help="How often checkpoints were saved (default: 100)")
args = parser.parse_args()

LOG_DIR   = os.path.expanduser(args.log_dir)
OUT_DIR   = os.path.expanduser(args.out_dir) if args.out_dir else LOG_DIR
CKPT_STEP = args.iters_per_ckpt

# ── Read TFEvents ─────────────────────────────────────────────────────────────
print(f"\nSearching for TFEvents in:\n  {LOG_DIR}\n")
tf_files = glob.glob(os.path.join(LOG_DIR, "events.out.tfevents.*"))
if not tf_files:
    # Also search one level down (some RSL-RL versions nest logs)
    tf_files = glob.glob(os.path.join(LOG_DIR, "**", "events.out.tfevents.*"),
                         recursive=True)
if not tf_files:
    print("ERROR: No TFEvents file found. Check --log_dir path.")
    sys.exit(1)

tf_path = sorted(tf_files)[-1]   # newest if multiple
print(f"Reading: {os.path.basename(tf_path)}")

try:
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator, STORE_EVERYTHING_SIZE_GUIDANCE)
    ea = EventAccumulator(
        tf_path,
        size_guidance={k: 0 for k in STORE_EVERYTHING_SIZE_GUIDANCE})
    ea.Reload()
    available_tags = ea.Tags().get("scalars", [])
    print(f"Available scalar tags ({len(available_tags)}):")
    for t in sorted(available_tags):
        print(f"  {t}")
    print()
except Exception as e:
    print(f"ERROR loading TFEvents: {e}")
    sys.exit(1)

# ── Extract scalars ───────────────────────────────────────────────────────────
# RSL-RL tag names (adjust if your logger uses different keys)
TAG_MAP = {
    # Internal name : possible TFBoard tag names (first match wins)
    # Ordered: most specific/version-specific first, generic fallbacks last
    "std":        ["Policy/mean_noise_std",          # ← your RSL-RL version
                   "Train/mean_noise_std",
                   "Action/mean_action_noise_std",
                   "action_noise_std",
                   "mean_noise_std"],
    "reward":     ["Train/mean_reward",              # ← confirmed present
                   "Reward/mean_reward",
                   "mean_reward"],
    "entropy":    ["Loss/entropy",                   # ← your RSL-RL version
                   "Train/mean_entropy_loss",
                   "Loss/entropy_loss",
                   "entropy_loss",
                   "mean_entropy_loss"],
    "ep_len":     ["Train/mean_episode_length",      # ← confirmed present
                   "Episode/mean_episode_length",
                   "mean_episode_length"],
    "value_loss": ["Loss/value_function",            # ← your RSL-RL version
                   "Train/mean_value_function_loss",
                   "Loss/value_function_loss",
                   "mean_value_function_loss"],
    "surr_loss":  ["Loss/surrogate",                 # ← your RSL-RL version
                   "Train/mean_surrogate_loss",
                   "Loss/surrogate_loss",
                   "mean_surrogate_loss"],
}

def pull(tag_candidates):
    """Return (steps[], values[]) for the first matching tag."""
    for tag in tag_candidates:
        if tag in available_tags:
            events = ea.Scalars(tag)
            steps  = np.array([e.step  for e in events], dtype=np.float32)
            vals   = np.array([e.value for e in events], dtype=np.float32)
            print(f"  ✓ Found tag: '{tag}'  ({len(steps)} points)")
            return steps, vals
    return None, None

print("Pulling metrics:")
std_s,   std_v   = pull(TAG_MAP["std"])
rew_s,   rew_v   = pull(TAG_MAP["reward"])
ent_s,   ent_v   = pull(TAG_MAP["entropy"])
eplen_s, eplen_v = pull(TAG_MAP["ep_len"])
vloss_s, vloss_v = pull(TAG_MAP["value_loss"])
sloss_s, sloss_v = pull(TAG_MAP["surr_loss"])
print()

if std_v is None or rew_v is None:
    print("ERROR: Could not find std or reward tags. "
          "Check that the tags above match your RSL-RL version.")
    sys.exit(1)

# ── Find available checkpoint files ──────────────────────────────────────────
ckpt_files = sorted(
    glob.glob(os.path.join(LOG_DIR, "model_*.pt")),
    key=lambda p: int(os.path.basename(p).replace("model_","").replace(".pt",""))
)
saved_iters = sorted(
    int(os.path.basename(p).replace("model_","").replace(".pt",""))
    for p in ckpt_files
)
print(f"Found {len(saved_iters)} checkpoint files  "
      f"(range: {saved_iters[0]} – {saved_iters[-1]})")

# ── Build a per-checkpoint metric table ───────────────────────────────────────
# RSL-RL logs one scalar per training iteration.
# model_N.pt = saved after iteration N → look up step N in the TFEvents.

def interp_at(steps, vals, target_iter):
    """Return metric value nearest to target_iter (±50 tolerance)."""
    if steps is None:
        return np.nan
    idx = np.argmin(np.abs(steps - target_iter))
    if abs(steps[idx] - target_iter) > 50:
        return np.nan
    return float(vals[idx])

rows = []
for it in saved_iters:
    std_val   = interp_at(std_s,   std_v,   it)
    rew_val   = interp_at(rew_s,   rew_v,   it)
    ent_val   = interp_at(ent_s,   ent_v,   it)
    eplen_val = interp_at(eplen_s, eplen_v, it)
    rows.append({
        "iter":    it,
        "std":     std_val,
        "reward":  rew_val,
        "entropy": ent_val,
        "ep_len":  eplen_val,
        "ckpt":    f"model_{it}.pt",
    })

rows = [r for r in rows if not np.isnan(r["std"])]

# ── Scoring function ──────────────────────────────────────────────────────────
# Deploy criteria (from training audit):
#   std     < 0.50   (hard gate — collapsed policy unusable)
#   entropy < 0.00   (negative = learning, positive = degenerate)
#   reward  maximise
#   ep_len  maximise (close to 999 = robot stays up)
#
# Score = reward_norm + ep_len_norm - std_penalty - entropy_penalty
# Negative std/entropy terms are heavy gates, not soft penalties.

def score(r):
    std     = r["std"]
    entropy = r["entropy"] if not np.isnan(r["entropy"]) else 0.0
    reward  = r["reward"]
    ep_len  = r["ep_len"] if not np.isnan(r["ep_len"]) else 0.0

    # Hard gates: collapsed policy scores very low
    if std > 2.0:
        return -1000.0 + reward * 0.001   # still rank them, but bottom tier
    if entropy > 5.0:
        return -500.0  + reward * 0.001

    # Soft score: normalised reward + normalised ep_len
    # Penalise rising std (even below 2.0, prefer lower)
    std_penalty     = std     * 10.0    # 0.20 std → -2.0 pts, 1.0 std → -10 pts
    entropy_penalty = max(0.0, entropy) * 5.0  # any positive entropy is bad

    return reward + ep_len * 0.03 - std_penalty - entropy_penalty

for r in rows:
    r["score"] = score(r)

rows_sorted = sorted(rows, key=lambda r: r["score"], reverse=True)

# ── Tier classification ───────────────────────────────────────────────────────
def tier(r):
    if r["std"] < 0.5 and (np.isnan(r["entropy"]) or r["entropy"] < 0):
        return "🟢 DEPLOY"
    elif r["std"] < 1.0 and (np.isnan(r["entropy"]) or r["entropy"] < 5):
        return "🟡 CHECK  "
    else:
        return "🔴 SKIP   "

# ── Print top candidates ──────────────────────────────────────────────────────
print("\n" + "="*90)
print(f"{'RANK':<5} {'ITER':<8} {'STD':<8} {'REWARD':<10} {'ENTROPY':<12} "
      f"{'EP_LEN':<10} {'SCORE':<10} {'STATUS'}")
print("="*90)

top_deploy = [r for r in rows_sorted if tier(r).startswith("🟢")][:5]
top_check  = [r for r in rows_sorted if tier(r).startswith("🟡")][:3]
shown = set()

rank = 1
for group_label, group in [("── Deploy candidates ──", top_deploy),
                            ("── Worth checking ──",    top_check)]:
    if group:
        print(f"\n  {group_label}")
    for r in group:
        if r["iter"] in shown:
            continue
        shown.add(r["iter"])
        print(f"  {rank:<5} {r['iter']:<8} "
              f"{r['std']:<8.3f} "
              f"{r['reward']:<10.2f} "
              f"{r['entropy'] if not np.isnan(r['entropy']) else 'N/A':<12.3f} "
              f"{r['ep_len'] if not np.isnan(r['ep_len']) else 'N/A':<10.1f} "
              f"{r['score']:<10.2f} "
              f"{tier(r)}")
        rank += 1

print("\n" + "="*90)

# ── Find collapse boundary ────────────────────────────────────────────────────
# Walk forward in time; find first iter where std > 1.0 and stays there
std_by_iter = {r["iter"]: r["std"] for r in rows}
collapse_iter = None
window = 5
iters_list = sorted(std_by_iter.keys())
for i in range(len(iters_list) - window):
    window_stds = [std_by_iter[iters_list[i+j]] for j in range(window)]
    if all(s > 1.0 for s in window_stds):
        collapse_iter = iters_list[i]
        break

if collapse_iter:
    print(f"\n  Collapse boundary detected at iter ≈ {collapse_iter}")
    last_healthy = max((it for it in iters_list if it < collapse_iter
                        and std_by_iter[it] < 0.5), default=None)
    if last_healthy:
        print(f"  Last healthy checkpoint before collapse: model_{last_healthy}.pt")
        print(f"  ★ PRIMARY RECOMMENDATION: model_{last_healthy}.pt")
    print()

# ── Save report ───────────────────────────────────────────────────────────────
report_path = os.path.join(OUT_DIR, "best_checkpoint_candidates.txt")
with open(report_path, "w") as f:
    f.write("Go1 Phase 3 — Checkpoint Ranking Report\n")
    f.write(f"Log dir: {LOG_DIR}\n\n")
    f.write(f"{'RANK':<5} {'ITER':<8} {'STD':<8} {'REWARD':<10} "
            f"{'ENTROPY':<12} {'EP_LEN':<10} {'SCORE':<10} STATUS\n")
    f.write("-"*80 + "\n")
    for rank_i, r in enumerate(rows_sorted[:30], 1):
        f.write(f"{rank_i:<5} {r['iter']:<8} "
                f"{r['std']:.3f}    "
                f"{r['reward']:.2f}      "
                f"{r['entropy'] if not np.isnan(r['entropy']) else float('nan'):.3f}         "
                f"{r['ep_len'] if not np.isnan(r['ep_len']) else float('nan'):.1f}      "
                f"{r['score']:.2f}      "
                f"{tier(r)}\n")
    if collapse_iter:
        f.write(f"\nCollapse boundary: iter {collapse_iter}\n")
        if last_healthy:
            f.write(f"★ PRIMARY RECOMMENDATION: model_{last_healthy}.pt\n")
print(f"Report saved: {report_path}")

# ── Plot ──────────────────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
    fig.suptitle("Go1 Phase 3 — Checkpoint Analysis", fontsize=13, y=0.98)

    iters_arr = np.array(iters_list)

    # ── Std ──
    ax = axes[0]
    std_arr = np.array([std_by_iter[i] for i in iters_list])
    ax.plot(iters_arr, std_arr, color="#378ADD", lw=1.2, label="action std")
    ax.axhline(0.5,  color="green",  ls="--", lw=1, label="deploy gate (0.5)")
    ax.axhline(1.0,  color="orange", ls="--", lw=1, label="check gate (1.0)")
    ax.axhline(2.0,  color="red",    ls="--", lw=1, label="collapse (2.0)")
    if collapse_iter:
        ax.axvline(collapse_iter, color="red", ls=":", lw=1.5, alpha=0.7,
                   label=f"collapse @{collapse_iter}")
    ax.set_ylabel("Action noise std")
    ax.set_ylim(0, min(std_arr.max() * 1.1, 15))
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    # ── Reward ──
    ax = axes[1]
    rew_arr = np.array([interp_at(rew_s, rew_v, i) for i in iters_list])
    ax.plot(iters_arr, rew_arr, color="#1D9E75", lw=1.2, label="mean reward")
    if collapse_iter:
        ax.axvline(collapse_iter, color="red", ls=":", lw=1.5, alpha=0.7)
    ax.set_ylabel("Mean reward")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Entropy ──
    ax = axes[2]
    if ent_v is not None:
        ent_arr = np.array([interp_at(ent_s, ent_v, i) for i in iters_list])
        ax.plot(iters_arr, ent_arr, color="#D85A30", lw=1.2, label="entropy loss")
        ax.axhline(0, color="black", ls="--", lw=0.8, label="healthy < 0")
        if collapse_iter:
            ax.axvline(collapse_iter, color="red", ls=":", lw=1.5, alpha=0.7)
        ax.set_ylabel("Entropy loss")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # ── Episode length ──
    ax = axes[3]
    if eplen_v is not None:
        ep_arr = np.array([interp_at(eplen_s, eplen_v, i) for i in iters_list])
        ax.plot(iters_arr, ep_arr, color="#7F77DD", lw=1.2, label="episode length")
        ax.axhline(990, color="green", ls="--", lw=1, label="ep=990 (good)")
        if collapse_iter:
            ax.axvline(collapse_iter, color="red", ls=":", lw=1.5, alpha=0.7)
        ax.set_ylabel("Episode length")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Highlight top deploy candidates on all axes
    for r in top_deploy[:3]:
        for ax in axes:
            ax.axvline(r["iter"], color="#27500A", ls="-", lw=0.8, alpha=0.5)

    axes[-1].set_xlabel("Training iteration")
    plt.tight_layout()

    plot_path = os.path.join(OUT_DIR, "best_checkpoint_analysis.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved:  {plot_path}")

except ImportError:
    print("matplotlib not available — plots skipped (report still saved)")

print("\nDone.\n")