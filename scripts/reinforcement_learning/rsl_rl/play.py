# play.py
# Standard OnPolicyRunner. Loads obs normalisation stats from checkpoint.
# Exports policy as JIT + ONNX for real Go1 deployment.

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Play a trained Go1 policy.")
parser.add_argument("--video",        action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs",     type=int, default=None)
parser.add_argument("--task",         type=str, default=None)
parser.add_argument("--agent",        type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed",         type=int, default=None)
parser.add_argument("--use_pretrained_checkpoint", action="store_true")
parser.add_argument("--real-time",    action="store_true", default=False)
parser.add_argument("--log",          action="store_true", default=False,
                    help="Save sim_log_<checkpoint>_<ts>.npz for sim-real comparison")
parser.add_argument("--log_steps",    type=int, default=1500,
                    help="Steps to record when --log is set (default 1500 = 30s at 50Hz)")
parser.add_argument("--phase2",       action="store_true", default=False,
                    help="Force Phase 2 delay DR U[0,8] per episode. REQUIRED for correct "
                         "evaluation of policies trained past _DELAY_PHASE1_END (step 15000). "
                         "Without this, delay=0 (Phase 1) — wrong conditions for Phase 2 policy.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch
import numpy as np
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg, RslRlVecEnvWrapper,
    export_policy_as_jit, export_policy_as_onnx,
)

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

from isaaclab_tasks.direct.go1.go1_env_cfg import Go1FlatEnvCfg
from isaaclab_tasks.direct.go1.agents.rsl_rl_ppo_cfg import Go1RslRlPpoCfg
try:
    from isaaclab_tasks.direct.go1.agents.rsl_rl_ppo_cfg import Go1SparsePPORunnerCfg
    _GO1_SPARSE_CFG_OK = True
except ImportError:
    Go1SparsePPORunnerCfg = None
    _GO1_SPARSE_CFG_OK = False

from isaaclab_tasks.direct.go2.go2_env_cfg import Go2FlatEnvCfg
from isaaclab_tasks.direct.go2.agents.rsl_rl_ppo_cfg import Go2RslRlPpoCfg
try:
    from isaaclab_tasks.direct.go2.agents.rsl_rl_ppo_cfg import Go2SparsePPORunnerCfg
    _GO2_SPARSE_CFG_OK = True
except ImportError:
    Go2SparsePPORunnerCfg = None
    _GO2_SPARSE_CFG_OK = False


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # ── Rebuild cfg from scratch for play — same reasoning as train.py:
    #    @configclass bakes scene(num_envs) at class definition time.
    is_go1 = "go1" in args_cli.task.lower()
    is_go2 = "go2" in args_cli.task.lower()
    is_sparse = "sparse" in args_cli.task.lower()
    is_rough = "rough" in args_cli.task.lower()
    is_sparse_rough = is_sparse and is_rough

    if is_go1:
        if is_sparse_rough:
            from isaaclab_tasks.direct.go1.go1_rough_env_cfg import (
                Go1RoughEnvCfg, make_rough_scene, TERRAIN_TOTAL_PATCHES)
            env_cfg = Go1RoughEnvCfg()
            agent_cfg = Go1SparsePPORunnerCfg() if _GO1_SPARSE_CFG_OK else Go1RslRlPpoCfg()
            print("[PLAY] Go1 SPARSE-ROUGH task")
        elif is_sparse:
            env_cfg = Go1FlatEnvCfg()
            agent_cfg = Go1SparsePPORunnerCfg() if _GO1_SPARSE_CFG_OK else Go1RslRlPpoCfg()
            print("[PLAY] Go1 SPARSE task")
        elif is_rough:
            from isaaclab_tasks.direct.go1.go1_rough_env_cfg import (
                Go1RoughEnvCfg, make_rough_scene, TERRAIN_TOTAL_PATCHES)
            env_cfg = Go1RoughEnvCfg()
            agent_cfg = Go1RslRlPpoCfg()
            print("[PLAY] Go1 ROUGH task")
        else:
            env_cfg = Go1FlatEnvCfg()
            agent_cfg = Go1RslRlPpoCfg()
            print("[PLAY] Go1 FLAT task")

    elif is_go2:
        if is_sparse_rough:
            from isaaclab_tasks.direct.go2.go2_rough_env_cfg import (
                Go2RoughEnvCfg, make_go2_rough_scene, TERRAIN_TOTAL_PATCHES)
            env_cfg = Go2RoughEnvCfg()
            agent_cfg = Go2SparsePPORunnerCfg() if _GO2_SPARSE_CFG_OK else Go2RslRlPpoCfg()
            print("[PLAY] Go2 SPARSE-ROUGH task")
        elif is_rough:
            from isaaclab_tasks.direct.go2.go2_rough_env_cfg import (
                Go2RoughEnvCfg, make_go2_rough_scene, TERRAIN_TOTAL_PATCHES)
            env_cfg = Go2RoughEnvCfg()
            agent_cfg = Go2RslRlPpoCfg()
            print("[PLAY] Go2 ROUGH task")
        else:
            env_cfg = Go2FlatEnvCfg()
            agent_cfg = Go2RslRlPpoCfg()
            print("[PLAY] Go2 FLAT task")

    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device

    # For play: 1 env is enough. Change num_envs class field in *_env_cfg.py
    # before running play if you want more. CLI --num_envs also works here
    # because we're just overriding scene.num_envs after fresh construction.
    if is_rough:
        # Rebuild scene to avoid @configclass num_envs freeze → stacking
        # make_rough_scene / make_go2_rough_scene were imported above under
        # the matching is_go1/is_go2 branch, whichever fired.
        _n = args_cli.num_envs if args_cli.num_envs is not None else TERRAIN_TOTAL_PATCHES
        if is_go1:
            env_cfg.scene = make_rough_scene(_n)
        else:
            env_cfg.scene = make_go2_rough_scene(_n)
        env_cfg.scene.num_envs = _n
        print(f"[PLAY] Rough scene: {_n} envs  ({TERRAIN_TOTAL_PATCHES} patches)")
    else:
        if args_cli.num_envs is not None:
            env_cfg.scene.num_envs = args_cli.num_envs
            env_cfg.scene.env_spacing = 4.0
            print(f"[PLAY] num_envs overridden to {args_cli.num_envs}")

    # ── Checkpoint path ───────────────────────────────────────────────────────
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    print(f"[PLAY] checkpoint : {resume_path}")
    print(f"[PLAY] num_envs   : {env_cfg.scene.num_envs}")

    # ── Create environment ────────────────────────────────────────────────────
    env = gym.make(args_cli.task, cfg=env_cfg,
                   render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        env = gym.wrappers.RecordVideo(env, **{
            "video_folder":   os.path.join(log_dir, "videos", "play"),
            "step_trigger":   lambda step: step == 0,
            "video_length":   args_cli.video_length,
            "disable_logger": True,
        })

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # ── Runner + custom load ──────────────────────────────────────────────────
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)

    def custom_load(self, path, load_optimizer=True):
        ckpt = torch.load(path, map_location=self.device)
        self.alg.policy.load_state_dict(ckpt["policy_state_dict"])
        # Restore normalizer — critical for correct inference on real Go1
        if "normalizer_state" in ckpt and ckpt["normalizer_state"] is not None:
            if hasattr(self.alg.policy, "actor_obs_normalizer"):
                self.alg.policy.actor_obs_normalizer.load_state_dict(ckpt["normalizer_state"])
                print(f"[LOAD] Normalizer restored")
                for k, v in ckpt["normalizer_state"].items():
                    if v.numel() <= 6:
                        print(f"  {k}: {v.cpu().numpy()}")
                    else:
                        print(f"  {k}[:5]: {v.flatten()[:5].cpu().numpy()}")
        else:
            print("[LOAD] WARNING: No normalizer_state in checkpoint!")
            print("  Policy will use untrained normalizer — may behave incorrectly on real robot")
        print(f"[LOAD] Policy restored from {path}")

    runner.load = custom_load.__get__(runner)
    runner.load(resume_path)
    print("[INFO] Checkpoint loaded.")

    # ── Optional sim logger ───────────────────────────────────────────────────
    # Enabled by --log flag. Captures same channels as real_log_*.npz from
    # go1_deploy_final.py so you can run compare_sim_real.py directly.
    robot_env = env.unwrapped.unwrapped  # Go1Env or Go2Env instance
    go1_env = robot_env  # keep alias for backward compat

    if args_cli.log:
        N = args_cli.log_steps
        # Tell go1_env how many steps to buffer and activate it
        go1_env._sim_log_maxsteps = N
        for k in go1_env._slog:               # resize pre-alloc arrays to N
            arr = go1_env._slog[k]
            go1_env._slog[k] = (np.zeros(N, arr.dtype) if arr.ndim == 1
                                 else np.zeros((N,) + arr.shape[1:], arr.dtype))
        go1_env._slog_step   = 0
        go1_env._slog_active = True           # go1_env.step() now writes buffers

        # Hook registered AFTER export below — torch.jit.script can't serialize hooks.
        _last_linear = None
        try:
            policy_nn_ref = runner.alg.policy
        except AttributeError:
            policy_nn_ref = runner.alg.actor_critic
        for m in policy_nn_ref.actor.modules():
            if isinstance(m, torch.nn.Linear):
                _last_linear = m
    else:
        go1_env._slog_active = False          # make sure it stays off
        _last_linear = None

    # ── Export policy for real robot ──────────────────────────────────────────
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    normalizer = getattr(policy_nn, "actor_obs_normalizer", None)
    export_dir = os.path.join(log_dir, "exported")
    export_policy_as_jit(policy_nn,  normalizer=normalizer, path=export_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_dir, filename="policy.onnx")
    print(f"[INFO] Exported to: {export_dir}")

    # ── Attach raw_net hook NOW (after JIT export — hook breaks torch.jit.script) ──
    if args_cli.log and _last_linear is not None:
        def _raw_net_hook(module, inp, out):
            s = go1_env._slog_step
            if go1_env._slog_active and s < go1_env._sim_log_maxsteps:
                go1_env._slog["raw_net"][s] = out[0].detach().cpu().numpy()
        _last_linear.register_forward_hook(_raw_net_hook)
        print(f"[LOG] raw_net hook attached  (steps to record: {args_cli.log_steps})")
    elif args_cli.log:
        print("[LOG] WARNING: could not find actor output layer — raw_net will be zeros")

    # ── Force training conditions ────────────────────────────────────────────
    # Policy was trained with:
    #   (a) obs noise ON  — do NOT disable, or distribution shift corrupts inference
    #   (b) Phase 2 delay DR (U[0,8]) — set _global_step to trigger Phase 2
    # go1_env._obs_noise_enabled = False  ← REMOVED: disabling noise is wrong for eval
    go1_env._obs_noise_enabled = True   # match training exactly

    if args_cli.phase2:
        if is_go2:
            from isaaclab_tasks.direct.go2.go2_env import (
                _DELAY_PHASE1_END, _DELAY_MAX)
        else:
            from isaaclab_tasks.direct.go1.go1_env import _DELAY_PHASE1_END
            _DELAY_MAX = 8
        go1_env._global_step = _DELAY_PHASE1_END
        go1_env._env_delays  = torch.randint(
            0, _DELAY_MAX + 1, (go1_env.num_envs,),
            device=go1_env.device, dtype=torch.long)
        print(f"[PLAY] Delays pre-sampled: mean={go1_env._env_delays.float().mean():.1f} "
              f"env0={go1_env._env_delays[0].item()} steps")
    else:
        print("[PLAY] WARNING: --phase2 not set → delay=0ms (Phase 1).")
        print("         Policy trained in Phase 2 (16ms DR) will be evaluated under")
        print("         wrong conditions. Use --phase2 for correct evaluation.")

    # ── Action range verification (printed before inference starts) ───────────
    print("[PLAY] Action range verification:")
    lo = go1_env._delta_soft_lo.cpu().numpy()
    hi = go1_env._delta_soft_hi.cpu().numpy()
    JNAMES = ["FL_hip","FR_hip","RL_hip","RR_hip",
              "FL_th","FR_th","RL_th","RR_th",
              "FL_kn","FR_kn","RL_kn","RR_kn"]
    print(f"  {'Joint':<10} {'lo':>7} {'hi':>7} {'mid':>7} {'half':>7}")
    for i, n in enumerate(JNAMES):
        mid  = (hi[i]+lo[i])/2
        half = (hi[i]-lo[i])/2
        print(f"  {n:<10} {lo[i]:>7.3f} {hi[i]:>7.3f} {mid:>7.3f} {half:>7.3f}")
    print()
    if is_go2:
        print("  [Deploy mapping] go2_rl_flat_1.py applies the tanh-squashed")
        print("  delta directly (no separate per-joint SCALE constants) — the")
        print("  delta_lo/hi above ARE the hardware-applied bounds, hip flip")
        print("  handled separately for FR_hip/RR_hip (see deploy script).")
    else:
        print("  [Deploy mapping] go1_deploy.py THIGH_SCALE=0.95, KNEE_SCALE=0.95, HIP_SCALE=0.70")
        print("  Max thigh hw delta = 0.35 × 0.95 = 0.332 rad (within Go1 URDF limits ✓)")
        print("  Max knee  hw delta = 0.35 × 0.95 = 0.332 rad (within Go1 URDF limits ✓)")
    print()

    policy   = runner.get_inference_policy(device=env.unwrapped.device)
    dt       = env.unwrapped.step_dt
    obs      = env.get_observations()
    timestep = 0

    print("[INFO] Running ...")
    while simulation_app.is_running():
        t0 = time.time()
        with torch.inference_mode():
            actions          = policy(obs)
            obs, _, _, _     = env.step(actions)

        if args_cli.video:
            timestep += 1
            if timestep >= args_cli.video_length:
                break

        # Stop when log buffer is full
        if args_cli.log and go1_env._slog_step >= args_cli.log_steps:
            print(f"[LOG] {args_cli.log_steps} steps recorded — stopping.")
            break

        sleep = dt - (time.time() - t0)
        if args_cli.real_time and sleep > 0:
            time.sleep(sleep)

    # ── Save sim log ──────────────────────────────────────────────────────────
    if args_cli.log:
        S        = go1_env._slog_step
        ckpt_tag = os.path.splitext(os.path.basename(resume_path))[0]  # e.g. model_24999
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(log_dir, "sim_logs")
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, f"sim_log_{ckpt_tag}_{ts}.npz")

        lo = go1_env._delta_soft_lo.cpu().numpy()
        hi = go1_env._delta_soft_hi.cpu().numpy()
        np.savez(out_path,
            obs_raw    = go1_env._slog["obs_raw"][:S],
            raw_net    = go1_env._slog["raw_net"][:S],
            tanh_delta = go1_env._slog["tanh_delta"][:S],
            target_q   = go1_env._slog["target_q"][:S],
            actual_q   = go1_env._slog["actual_q"][:S],
            actual_qd  = go1_env._slog["actual_qd"][:S],
            proj_grav  = go1_env._slog["proj_grav"][:S],
            ang_vel    = go1_env._slog["ang_vel"][:S],
            lin_vel    = go1_env._slog["lin_vel"][:S],
            cmd        = go1_env._slog["cmd"][:S],
            contact    = go1_env._slog["contact"][:S],
            tilt_deg   = go1_env._slog["tilt_deg"][:S],
            reward     = go1_env._slog["reward"][:S],
            default_q  = go1_env._robot.data.default_joint_pos[0].cpu().numpy(),
            delta_lo   = lo,
            delta_hi   = hi,
            step_dt    = np.array([env.unwrapped.step_dt]),
            src        = np.array(["sim"], dtype=object),
            checkpoint = np.array([resume_path], dtype=object),
        )

        tilt  = go1_env._slog["tilt_deg"][:S]
        lv    = go1_env._slog["lin_vel"][:S, 0]
        td    = go1_env._slog["tanh_delta"][:S]
        NAMES = ['FL_hip','FR_hip','RL_hip','RR_hip']
        print(f"\n[LOG] Saved {S} steps → {out_path}")
        print(f"  tilt   : mean={tilt.mean():.1f}°  max={tilt.max():.1f}°  "
              f">20°:{(tilt>20).mean()*100:.0f}%")
        print(f"  lv_x   : mean={lv.mean():.3f} m/s")
        print(f"  raw_net: [{go1_env._slog['raw_net'][:S].min():.1f}, "
              f"{go1_env._slog['raw_net'][:S].max():.1f}]")
        print(f"  hip sat (at lo limit):")
        for i, n in enumerate(NAMES):
            sat = (td[:, i] <= lo[i] * 0.98).mean() * 100
            print(f"    {n}: {sat:.0f}%  {'*** saturated' if sat > 30 else 'ok'}")
        print(f"\n  Compare with real robot:")
        print(f"  python compare_sim_real.py {out_path} <real_log_*.npz>")

    env.close()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
    simulation_app.close()