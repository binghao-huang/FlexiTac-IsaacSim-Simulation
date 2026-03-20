# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.utils import configclass

from ..sensor_base_cfg import SensorBaseCfg
from .warp_sdf_tactile_sensor import WarpSdfTactileSensor


@configclass
class WarpSdfTactileSensorCfg(SensorBaseCfg):
    """Configuration for :class:`WarpSdfTactileSensor`.

    This is a minimal Warp-based tactile sensor intended for testing a VT-refine-like
    taxel grid pipeline with an analytic oriented-box SDF.

    Output matches VT-refine's `tactile_points_w` contract: (E, S*P, 4) per step.
    """

    class_type: type = WarpSdfTactileSensor

    # Sensor attachment
    elastomer_prim_paths: list[str] = list()

    # Taxel grid
    num_rows: int = 12
    num_cols: int = 32
    point_distance: float = 0.002
    normal_axis: int = 0
    normal_offset: float = -0.0032

    # Optional patch pose in body frame
    patch_offset_pos_b: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Quaternion is wxyz.
    patch_offset_quat_b: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    patch_offset_pos_b_per_elastomer: list[tuple[float, float, float]] | None = None
    patch_offset_quat_b_per_elastomer: list[tuple[float, float, float, float]] | None = None

    # Analytic target SDF: oriented box in world frame
    box_pos_w: tuple[float, float, float] = (0.6, 0.0, 0.55)
    box_quat_w: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    box_half_extents: tuple[float, float, float] = (0.04, 0.04, 0.04)

    # If `target_mesh_prim_path` is provided, the sensor will query distance to the mesh surface using Warp.
    # - Signed distance requires a watertight mesh (Warp provides `sign` for closed meshes).
    # - If `mesh_use_signed_distance` is False, the sensor uses unsigned distance and a small `mesh_shell_thickness` to emulate penetration.
    target_mesh_prim_path: str | None = None
    mesh_max_dist: float = 0.20
    mesh_use_signed_distance: bool = True
    mesh_signed_distance_method: str = "winding"
    # When `mesh_signed_distance_method == "normal"`, use vertex-normal interpolation (Phong-like)
    # instead of flat triangle normals.
    mesh_smooth_normals: bool = True
    mesh_shell_thickness: float = 0.0015

    # Simple spring law: fn = clamp(k * max(-sdf, 0), 0, max_force)
    stiffness: float = 5_000.0
    max_force: float = 2.4

    # Normalize output fn into [0,1] by dividing by max_force
    normalize_forces: bool = True

    # Debug visualization (only used when SensorBaseCfg.debug_vis=True)
    debug_vis_env_id: int = 0
    debug_vis_point_radius: float = 0.002
    debug_vis_force_threshold: float = 1.0e-6
    debug_vis_show_all_taxels: bool = False
    debug_vis_show_axes: bool = False
    debug_vis_axes_scale: float = 0.05
