"""
Minimal example: replay a vt-refine trajectory in ALOHA tactile env + visualize tactile grids.

Usage:
    cd ~/repos/IsaacLab
    ./isaaclab.sh -p Isaacsim_tactile_env/reply_with_tactile.py \
      --dataset_npz Isaacsim_tactile_env/data/dataset_train.npz \
      --normalization_pth Isaacsim_tactile_env/data/dataset_normalizer.npz \
      --episode_idx 0 \
      --replay_key joint_states \
      --steps_per_frame 3
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# --- SimulationApp MUST be created before any omni/isaaclab imports ---
from isaaclab.app import AppLauncher


# -----------------------------
# Args (minimal)
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_npz", type=str, required=True)
parser.add_argument("--normalization_pth", type=str, default=None)
parser.add_argument("--episode_idx", type=int, default=0)
parser.add_argument("--replay_key", type=str, default="joint_states", choices=("joint_states", "actions"))
parser.add_argument("--steps_per_frame", type=int, default=3)
parser.add_argument("--max_steps", type=int, default=-1)
parser.add_argument("--tactile_scale", type=int, default=8)

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# --- Now safe to import env ---
sys.path.insert(0, str(Path(__file__).parent))
from aloha_tactile_env import AlohaTactileEnv, AlohaTactileEnvCfg  # noqa: E402


SENSOR_LABELS = ["L-L", "L-R", "R-L", "R-R"]


# -----------------------------
# Dataset utilities (minimal)
# -----------------------------
def _denorm(x: np.ndarray, x_min: np.ndarray, x_max: np.ndarray) -> np.ndarray:
    return (x + 1.0) * 0.5 * (x_max - x_min) + x_min


def load_episode(npz_path: str, norm_path: str | None, *, key: str, episode_idx: int) -> np.ndarray:
    npz_path = os.path.expanduser(npz_path)
    data = np.load(npz_path, allow_pickle=False)

    if key not in data:
        raise KeyError(f"Key '{key}' not in dataset. Have: {list(data.keys())}")
    if "traj_lengths" not in data:
        raise KeyError("Dataset missing 'traj_lengths'")

    vals = data[key].astype(np.float32)
    lengths = np.asarray(data["traj_lengths"], dtype=np.int64)
    if not (0 <= episode_idx < lengths.size):
        raise ValueError(f"episode_idx={episode_idx} out of range (num_episodes={lengths.size})")

    starts = np.concatenate(([0], np.cumsum(lengths)[:-1]))
    s = int(starts[episode_idx])
    L = int(lengths[episode_idx])
    ep = vals[s : s + L]

    if norm_path is None:
        return ep

    stats = torch.load(os.path.expanduser(norm_path), map_location="cpu")
    min_key = f"stats.{key}.min"
    max_key = f"stats.{key}.max"
    if min_key not in stats or max_key not in stats:
        raise KeyError(f"Normalization missing {min_key}/{max_key}")

    x_min = stats[min_key].detach().cpu().numpy().astype(np.float32)
    x_max = stats[max_key].detach().cpu().numpy().astype(np.float32)
    return _denorm(ep, x_min, x_max).astype(np.float32)

def show_image(name: str, img_bgr: np.ndarray) -> None:
    """Show image with OpenCV HighGUI (no dump)."""
    if img_bgr is None:
        return
    if img_bgr.dtype != np.uint8:
        img_bgr = np.clip(img_bgr, 0, 255).astype(np.uint8)

    cv2.imshow(name, img_bgr)
    cv2.waitKey(1)  

def tactile_strip(tactile: np.ndarray, *, scale: int, labels: list[str]) -> np.ndarray:
    """tactile: (4,H,W) -> BGR strip"""
    imgs = []
    for i in range(min(4, tactile.shape[0])):
        grid = tactile[i]
        vmax = float(grid.max())
        norm = (grid / (vmax + 1e-8)) if vmax > 0 else np.zeros_like(grid)
        u8 = (norm * 255.0).astype(np.uint8)
        bgr = cv2.applyColorMap(u8, cv2.COLORMAP_JET)
        if scale != 1:
            bgr = cv2.resize(bgr, (bgr.shape[1] * scale, bgr.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
        cv2.putText(
            bgr,
            f"{labels[i]} max={vmax:.3f}",
            (5, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        imgs.append(bgr)
    return np.concatenate(imgs, axis=1) if imgs else np.zeros((1, 1, 3), dtype=np.uint8)


# -----------------------------
# Main
# -----------------------------
def main():
    # normalization auto-guess (optional)
    norm = args.normalization_pth
    if norm is None:
        guess = str(Path(os.path.expanduser(args.dataset_npz)).parent / "normalization.pth")
        if os.path.isfile(guess):
            norm = guess

    traj = load_episode(args.dataset_npz, norm, key=args.replay_key, episode_idx=args.episode_idx)
    ep_len = traj.shape[0]
    print(f"[INFO] episode={args.episode_idx} key={args.replay_key} len={ep_len} shape={traj.shape}")

    # env config: keep minimal + match AppLauncher flags
    cfg = AlohaTactileEnvCfg(
        headless=bool(getattr(args, "headless", True)),
        device=str(getattr(args, "device", "cuda:0")),
        enable_camera=bool(getattr(args, "enable_cameras", False)),
    )

    if not cfg.headless:
        try:
            from isaacsim.core.utils.viewports import set_camera_view
            set_camera_view(list(cfg.camera_eye), list(cfg.camera_target))
        except Exception as e:
            print(f"[WARN] Failed to set viewport camera view: {e}")

    env = AlohaTactileEnv(cfg, simulation_app=simulation_app)
    env.reset()

    spf = max(1, int(args.steps_per_frame))
    total = ep_len * spf
    if args.max_steps > 0:
        total = min(total, int(args.max_steps))

    for step in range(total):
        if not simulation_app.is_running():
            break

        traj_idx = min(step // spf, ep_len - 1)
        action = traj[traj_idx].copy()

        # keep your small tweak (optional)
        if action.shape[0] >= 16:
            action[14] -= 0.005
            action[15] += 0.005

        obs, *_ = env.step(action)
        tactile = obs["tactile"]  # (4,H,W)
        vis = tactile_strip(tactile, scale=int(args.tactile_scale), labels=SENSOR_LABELS)
        show_image("Tactile", vis)

        if step % 30 == 0:
            print(f"[step {step:06d}] traj_idx={traj_idx:4d} tactile_max={float(tactile.max()):.4f}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
