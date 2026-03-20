# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import gymnasium
import numpy as np
import torch
from pathlib import Path

@dataclass
class TrackInfo:
    rp: object          # RigidPrim instance
    p_rel: torch.Tensor # (3,) target position in RB frame
    q_rel: torch.Tensor # (4,) target quat in RB frame
    rb_path: str


@dataclass
class AlohaTactileEnvCfg:
    """Configuration for the ALOHA tactile inference environment."""

    # Robot
    urdf_path: str = Path(__file__).resolve().parent / "assets" / "aloha_tactile.urdf"
    robot_prim_path: str = "/World/Robot"
    fix_base: bool = False
    merge_fixed_joints: bool = False
    urdf_drive_stiffness: float = 400.0
    urdf_drive_damping: float = 40.0
    force_urdf_conversion: bool = True
    usd_output_dir: str | None = None

    # Objects
    enable_plug: bool = True
    enable_socket: bool = True
    asset_root: str = Path(__file__).resolve().parent / "assets"
    automate_asset_id: str = "00186"
    plug_fix_base: bool = False
    socket_fix_base: bool = False
    plug_scale: float = 1.06
    socket_scale: float = 1.0
    plug_collider_type: str = "convex_decomposition"
    socket_collider_type: str = "convex_decomposition"
    force_objects_urdf_conversion: bool = True
    plug_default_pose = (0.0, +0.05, +0.003, 0.0, 0.0, +1.0, 0.0)
    socket_default_pose = (0.0, -0.05, +0.003, 0.0, 0.0, +1.0, 0.0)

    # Tactile sensor
    num_rows: int = 12
    num_cols: int = 32
    point_distance: float = 0.002
    normal_axis: int = 0
    normal_offset: float = 0.0036
    patch_offset_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    patch_offset_quat: tuple[float, float, float, float] = (0.7071068, 0.0, 0.0, -0.7071068)
    stiffness: float = 5_000.0
    max_force: float = 10.0
    mesh_max_dist: float = 0.20
    mesh_signed: bool = True
    mesh_signed_distance_method: str = "winding"
    mesh_shell_thickness: float = 0.001
    left_arm_target_mesh_prim: str = "/World/Socket"
    right_arm_target_mesh_prim: str = "/World/Plug"
    debug_vis: bool = False

    # Camera / rendering
    enable_camera: bool = True
    camera_width: int = 640
    camera_height: int = 480
    camera_prim_path: str = "/World/Camera"
    camera_eye: tuple[float, float, float] = (-0.9, 0.0, 0.4)
    camera_target: tuple[float, float, float] = (0.0, 0.0, 0.1)
    save_renders: bool = False
    render_output_dir: str = ""

    # Simulation
    physics_dt: float = 1.0 / 120.0
    device: str = "cuda:0"
    headless: bool = True


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_JOINT_ORDER = [
    "left/waist", "left/shoulder", "left/elbow", "left/forearm_roll",
    "left/wrist_angle", "left/wrist_rotate", "left/left_finger", "left/right_finger",
    "right/waist", "right/shoulder", "right/elbow", "right/forearm_roll",
    "right/wrist_angle", "right/wrist_rotate", "right/left_finger", "right/right_finger",
]