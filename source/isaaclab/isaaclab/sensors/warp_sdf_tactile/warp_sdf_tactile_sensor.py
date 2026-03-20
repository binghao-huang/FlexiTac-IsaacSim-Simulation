# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# pyright: ignore

from collections.abc import Sequence

import torch

from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_mul

from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from ..contact_sensor import ContactSensor, ContactSensorCfg
from ..sensor_base import SensorBase
from .warp_sdf_tactile_data import WarpSdfTactileSensorData

import warp as wp  # type: ignore

from isaaclab.utils.warp.ops import convert_to_warp_mesh


@wp.kernel(enable_backward=False)
def mesh_distance_kernel(
    queries_l: wp.array(dtype=wp.vec3),
    mesh: wp.uint64,
    tri_indices: wp.array(dtype=wp.int32),
    vertex_normals: wp.array(dtype=wp.vec3),
    max_dist: float,
    signed_mode: int,
    smooth_normals: int,
    dist_out: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    q = queries_l[tid]

    sign = float(0.0)
    face_idx = int(0)
    face_u = float(0.0)
    face_v = float(0.0)

    hit = wp.mesh_query_point(mesh, q, max_dist, sign, face_idx, face_u, face_v)
    if hit:
        p = wp.mesh_eval_position(mesh, face_idx, face_u, face_v)
        delta = q - p
        d = wp.length(delta)

        if signed_mode == 0:
            # unsigned distance
            dist_out[tid] = d
        elif signed_mode == 1:
            # Warp's winding-number based sign (requires watertight mesh)
            dist_out[tid] = sign * d
        else:
            # Normal-based sign (works for open meshes but depends on consistent normals)
            # Compute triangle normal (flat)
            p0 = wp.mesh_eval_position(mesh, face_idx, 0.0, 0.0)
            p1 = wp.mesh_eval_position(mesh, face_idx, 1.0, 0.0)
            p2 = wp.mesh_eval_position(mesh, face_idx, 0.0, 1.0)
            n_face = wp.cross(p1 - p0, p2 - p0)
            n_face_len = wp.length(n_face)
            if n_face_len > 1.0e-12:
                n_face = n_face / n_face_len
            else:
                n_face = wp.vec3(0.0, 0.0, 1.0)

            n = n_face
            if smooth_normals == 1:
                # Interpolate vertex normals using barycentric coords
                base = face_idx * 3
                i0 = tri_indices[base + 0]
                i1 = tri_indices[base + 1]
                i2 = tri_indices[base + 2]
                w0 = 1.0 - face_u - face_v
                w1 = face_u
                w2 = face_v
                n_interp = w0 * vertex_normals[i0] + w1 * vertex_normals[i1] + w2 * vertex_normals[i2]
                n_len = wp.length(n_interp)
                if n_len > 1.0e-12:
                    n_interp = n_interp / n_len
                else:
                    n_interp = n_face
                # Align interpolated normal with the triangle's orientation
                if wp.dot(n_interp, n_face) < 0.0:
                    n_interp = -n_interp
                n = n_interp

            sd = wp.dot(delta, n)
            s = float(1.0)
            if sd < 0.0:
                s = float(-1.0)
            dist_out[tid] = s * d
    else:
        dist_out[tid] = max_dist


@wp.func
def _quat_rotate_inv(q: wp.vec4, v: wp.vec3) -> wp.vec3:
    # q is (w, x, y, z). Compute inverse rotation by using conjugate.
    qw = q[0]
    qx = q[1]
    qy = q[2]
    qz = q[3]
    # conjugate
    cx = -qx
    cy = -qy
    cz = -qz
    # quat * v
    # treat v as pure quaternion (0, v)
    tx = qw * v[0] + cy * v[2] - cz * v[1]
    ty = qw * v[1] + cz * v[0] - cx * v[2]
    tz = qw * v[2] + cx * v[1] - cy * v[0]
    tw = -cx * v[0] - cy * v[1] - cz * v[2]
    # result = (t) * conj(q)
    rx = tw * cx + tx * qw + ty * cz - tz * cy
    ry = tw * cy - tx * cz + ty * qw + tz * cx
    rz = tw * cz + tx * cy - ty * cx + tz * qw
    return wp.vec3(rx, ry, rz)


@wp.kernel(enable_backward=False)
def box_sdf_kernel(
    points_w: wp.array(dtype=wp.vec3),
    box_pos_w: wp.vec3,
    box_quat_w: wp.vec4,
    half_extents: wp.vec3,
    sdf_out: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    p_w = points_w[tid]
    # transform point into box local frame
    p_l = _quat_rotate_inv(box_quat_w, p_w - box_pos_w)

    qx = wp.abs(p_l.x) - half_extents.x
    qy = wp.abs(p_l.y) - half_extents.y
    qz = wp.abs(p_l.z) - half_extents.z

    # outside distance
    ox = wp.max(qx, 0.0)
    oy = wp.max(qy, 0.0)
    oz = wp.max(qz, 0.0)
    outside = wp.sqrt(ox * ox + oy * oy + oz * oz)

    # inside distance (negative)
    m = wp.max(qx, qy)
    m = wp.max(m, qz)
    inside = wp.min(m, 0.0)

    sdf_out[tid] = outside + inside


class WarpSdfTactileSensor(SensorBase):
    """Warp-based tactile sensor using an analytic oriented-box SDF.

    This is a test-oriented sensor to validate a VT-refine-like taxel pipeline in Isaac Lab
    without relying on PhysX contact forces.

    Output tensors match the VT-refine contract: `tactile_points_w` with [x,y,z,fn].
    """

    def __init__(self, cfg):
        super().__init__(cfg)

        if not getattr(self.cfg, "elastomer_prim_paths", None):
            raise ValueError("'elastomer_prim_paths' must be a non-empty list")

        if self.cfg.num_rows <= 0 or self.cfg.num_cols <= 0:
            raise ValueError("'num_rows' and 'num_cols' must be positive")
        if self.cfg.point_distance <= 0.0:
            raise ValueError("'point_distance' must be positive")
        if self.cfg.normal_axis not in (0, 1, 2):
            raise ValueError("'normal_axis' must be one of 0, 1, 2")

        if self.cfg.stiffness <= 0.0:
            raise ValueError("'stiffness' must be positive")
        if self.cfg.max_force <= 0.0:
            raise ValueError("'max_force' must be positive")

        self._data = WarpSdfTactileSensorData()

        self._points_local_per_sensor: torch.Tensor | None = None
        self._num_points: int = 0

        # Track pose via lightweight contact sensors (pose only).
        self._pose_sensors: list[ContactSensor] = []
        for elastomer_prim_path in self.cfg.elastomer_prim_paths:
            pose_cfg = ContactSensorCfg(
                prim_path=elastomer_prim_path,
                update_period=self.cfg.update_period,
                history_length=0,
                debug_vis=False,
                track_pose=True,
                track_contact_points=False,
                filter_prim_paths_expr=[],
            )
            self._pose_sensors.append(ContactSensor(pose_cfg))

        self._num_sensors = len(self._pose_sensors)

        # Box pose (world). Initialized on simulator PLAY when device is available.
        self._box_pos_w: torch.Tensor | None = None
        self._box_quat_w: torch.Tensor | None = None
        self._box_half_extents: torch.Tensor | None = None

        # Optional mesh target (loaded from USD).
        self._target_mesh_prim_path: str | None = getattr(self.cfg, "target_mesh_prim_path", None)
        self._wp_mesh: wp.Mesh | None = None
        self._wp_mesh_tri_indices: wp.array | None = None
        self._wp_mesh_vertex_normals: wp.array | None = None
        self._mesh_pos_w: torch.Tensor | None = None
        self._mesh_quat_w: torch.Tensor | None = None
        self._mesh_scale_w: torch.Tensor | None = None

        # Warp device (initialized on PLAY).
        self._wp_device: str | None = None

        # Reusable buffers for warp outputs
        self._sdf_out: torch.Tensor | None = None

        # Debug visualization
        self._debug_markers: VisualizationMarkers | None = None
        self._debug_axes: VisualizationMarkers | None = None

    @property
    def data(self) -> WarpSdfTactileSensorData:
        self._update_outdated_buffers()
        return self._data

    def set_box_pose(
        self,
        pos_w: tuple[float, float, float] | torch.Tensor,
        quat_w: tuple[float, float, float, float] | torch.Tensor,
    ):
        """Update the target box pose in world frame."""
        device = self.device
        if isinstance(pos_w, torch.Tensor):
            self._box_pos_w = pos_w.to(device=device, dtype=torch.float32)
        else:
            self._box_pos_w = torch.tensor(pos_w, device=device, dtype=torch.float32)

        if isinstance(quat_w, torch.Tensor):
            self._box_quat_w = quat_w.to(device=device, dtype=torch.float32)
        else:
            self._box_quat_w = torch.tensor(quat_w, device=device, dtype=torch.float32)

    def set_target_pose(
        self,
        pos_w: tuple[float, float, float] | torch.Tensor,
        quat_w: tuple[float, float, float, float] | torch.Tensor,
    ):
        """Update the target mesh pose in world frame (used when `target_mesh_prim_path` is set)."""
        device = self.device
        if isinstance(pos_w, torch.Tensor):
            self._mesh_pos_w = pos_w.to(device=device, dtype=torch.float32)
        else:
            self._mesh_pos_w = torch.tensor(pos_w, device=device, dtype=torch.float32)

        if isinstance(quat_w, torch.Tensor):
            self._mesh_quat_w = quat_w.to(device=device, dtype=torch.float32)
        else:
            self._mesh_quat_w = torch.tensor(quat_w, device=device, dtype=torch.float32)

        # Cache mesh scale once if not available. This is important for scaled USD meshes
        # (e.g., unit cube mesh with a scale xform to represent size), otherwise distances are wrong.
        if self._mesh_scale_w is None and self._target_mesh_prim_path is not None:
            try:
                prim = self.stage.GetPrimAtPath(self._target_mesh_prim_path)
                if prim.IsValid():
                    sx, sy, sz = sim_utils.resolve_prim_scale(prim)
                    self._mesh_scale_w = torch.tensor((sx, sy, sz), device=device, dtype=torch.float32)
            except (ValueError, RuntimeError):
                pass

    def reset(self, env_ids: Sequence[int] | None = None):
        super().reset(env_ids)
        for s in self._pose_sensors:
            s.reset(env_ids)

        resolved_env_ids = slice(None) if env_ids is None else env_ids
        if self._data.tactile_points_w is not None:
            self._data.tactile_points_w[resolved_env_ids] = 0.0
        if self._data.tactile_points_w_per_sensor is not None:
            self._data.tactile_points_w_per_sensor[resolved_env_ids] = 0.0

    def _initialize_impl(self):
        super()._initialize_impl()

        # Initialize box tensors and warp device now that SensorBase has a valid device.
        if self._box_pos_w is None:
            self._box_pos_w = torch.tensor(self.cfg.box_pos_w, device=self.device, dtype=torch.float32)
        if self._box_quat_w is None:
            self._box_quat_w = torch.tensor(self.cfg.box_quat_w, device=self.device, dtype=torch.float32)
        if self._box_half_extents is None:
            self._box_half_extents = torch.tensor(self.cfg.box_half_extents, device=self.device, dtype=torch.float32)
        if self._wp_device is None:
            dev = self.device
            # Isaac Lab uses device strings like "cuda:0"/"cpu".
            if dev.startswith("cuda"):
                self._wp_device = dev if ":" in dev else "cuda:0"
            else:
                self._wp_device = "cpu"

        # If a target mesh prim path is provided, load the USD mesh and build a Warp mesh (triangles).
        if self._target_mesh_prim_path is not None and self._wp_mesh is None:
            assert self._wp_device is not None
            (
                self._wp_mesh,
                self._wp_mesh_tri_indices,
                self._wp_mesh_vertex_normals,
            ) = self._load_warp_mesh_from_usd(self._target_mesh_prim_path, device=self._wp_device)

            # Initialize target pose from the USD prim.
            try:
                prim = self.stage.GetPrimAtPath(self._target_mesh_prim_path)
                if prim.IsValid():
                    pos, quat = sim_utils.resolve_prim_pose(prim)
                    self.set_target_pose(pos, quat)
                    # cache scale for correct world->local transform when querying distances
                    sx, sy, sz = sim_utils.resolve_prim_scale(prim)
                    self._mesh_scale_w = torch.tensor((sx, sy, sz), device=self.device, dtype=torch.float32)
            except (ValueError, RuntimeError):
                # If pose resolution fails, user can still drive pose via `set_target_pose`.
                pass

        base_points_local = self._create_local_grid_points(
            num_rows=self.cfg.num_rows,
            num_cols=self.cfg.num_cols,
            point_distance=self.cfg.point_distance,
            normal_axis=self.cfg.normal_axis,
            normal_offset=self.cfg.normal_offset,
            device=self._device,
        )
        self._num_points = int(base_points_local.shape[0])

        offset_pos_list = self._resolve_patch_offset_pos_list()
        offset_quat_list = self._resolve_patch_offset_quat_list()

        points_local_per_sensor = []
        for offset_pos, offset_quat in zip(offset_pos_list, offset_quat_list, strict=True):
            pos_b = torch.tensor(offset_pos, device=self._device, dtype=torch.float32)
            quat_b = torch.tensor(offset_quat, device=self._device, dtype=torch.float32)
            pts = quat_apply(quat_b.unsqueeze(0).expand(self._num_points, -1), base_points_local) + pos_b
            points_local_per_sensor.append(pts)
        self._points_local_per_sensor = torch.stack(points_local_per_sensor, dim=0)  # (S, P, 3)

        self._data.tactile_points_w_per_sensor = torch.zeros(
            (self._num_envs, self._num_sensors, self._num_points, 4), device=self._device, dtype=torch.float32
        )
        self._data.tactile_points_w = torch.zeros(
            (self._num_envs, self._num_sensors * self._num_points, 4), device=self._device, dtype=torch.float32
        )

        # Allocate sdf output buffer: per env, per sensor, per point.
        self._sdf_out = torch.empty(
            (self._num_envs, self._num_sensors, self._num_points), device=self._device, dtype=torch.float32
        )

        if self.cfg.debug_vis:
            radius = float(getattr(self.cfg, "debug_vis_point_radius", 0.002))
            show_all = bool(getattr(self.cfg, "debug_vis_show_all_taxels", False))
            # Two prototypes: contact (visible) and no_contact (hidden).
            vis_cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/WarpSdfTactile",
                markers={
                    "contact": sim_utils.SphereCfg(
                        radius=radius,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                    ),
                    "no_contact": sim_utils.SphereCfg(
                        radius=radius,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.4, 1.0)),
                        visible=show_all,
                    ),
                },
            )
            self._debug_markers = VisualizationMarkers(vis_cfg)

            # Optional: show XYZ axes for each attached elastomer body.
            if bool(getattr(self.cfg, "debug_vis_show_axes", False)):
                axes_scale = float(getattr(self.cfg, "debug_vis_axes_scale", 0.05))
                axes_cfg = VisualizationMarkersCfg(
                    prim_path="/Visuals/WarpSdfTactileAxes",
                    markers={
                        "frame": sim_utils.UsdFileCfg(
                            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                            scale=(axes_scale, axes_scale, axes_scale),
                        )
                    },
                )
                self._debug_axes = VisualizationMarkers(axes_cfg)






    def _update_buffers_impl(self, env_ids: Sequence[int]):
        assert self._points_local_per_sensor is not None
        assert self._data.tactile_points_w_per_sensor is not None
        assert self._data.tactile_points_w is not None
        assert self._sdf_out is not None
        use_mesh = self._target_mesh_prim_path is not None and self._wp_mesh is not None
        if use_mesh:
            assert self._mesh_pos_w is not None
            assert self._mesh_quat_w is not None
        else:
            assert self._box_pos_w is not None
            assert self._box_quat_w is not None
            assert self._box_half_extents is not None
        assert self._wp_device is not None

        # Keep internal pose sensors fresh.
        for s in self._pose_sensors:
            if s.is_initialized:
                s.update(0.0, force_recompute=True)

        num_envs = int(env_ids.numel()) if isinstance(env_ids, torch.Tensor) else len(env_ids)

        tactile_per_sensor = []
        sdf_per_sensor = []

        for sensor_idx, s in enumerate(self._pose_sensors):
            if not s.is_initialized:
                tactile_per_sensor.append(
                    torch.zeros((num_envs, self._num_points, 4), device=self._device, dtype=torch.float32)
                )
                sdf_per_sensor.append(torch.full((num_envs, self._num_points), float("inf"), device=self._device))
                continue

            if s.num_bodies != 1:
                raise RuntimeError(
                    "WarpSdfTactileSensor expects one rigid body per elastomer prim path. "
                    f"Got {s.num_bodies} bodies for prim_path='{s.cfg.prim_path}'."
                )

            sd = s.data
            assert sd.pos_w is not None
            assert sd.quat_w is not None

            pos_w = sd.pos_w[env_ids, 0]  # (E, 3)
            quat_w = sd.quat_w[env_ids, 0]  # (E, 4)

            points_local = self._points_local_per_sensor[sensor_idx].unsqueeze(0).expand(num_envs, -1, -1)
            quat = quat_w.unsqueeze(1).expand(-1, self._num_points, -1)
            points_w = pos_w.unsqueeze(1) + quat_apply(quat, points_local)  # (E, P, 3)

            # Warp distance query (box analytic SDF or USD mesh).
            sdf_e = torch.empty((num_envs, self._num_points), device=self._device, dtype=torch.float32)

            if use_mesh:
                wp_mesh = self._wp_mesh
                assert wp_mesh is not None
                tri_indices = self._wp_mesh_tri_indices
                vertex_normals = self._wp_mesh_vertex_normals
                if tri_indices is None or vertex_normals is None:
                    raise RuntimeError("Mesh query requested but mesh auxiliary buffers are not initialized.")

                mesh_pos = self._mesh_pos_w
                mesh_quat = self._mesh_quat_w
                assert mesh_pos is not None and mesh_quat is not None
                mesh_scale = self._mesh_scale_w

                max_dist = float(getattr(self.cfg, "mesh_max_dist", 0.20))

                use_signed = bool(getattr(self.cfg, "mesh_use_signed_distance", False))
                signed_method = str(getattr(self.cfg, "mesh_signed_distance_method", "winding")).lower()
                if not use_signed:
                    signed_mode = 0
                else:
                    signed_mode = 2 if signed_method == "normal" else 1
                smooth_normals = int(bool(getattr(self.cfg, "mesh_smooth_normals", True)))

                for e in range(num_envs):
                    pts_w = points_w[e]
                    pts_l = quat_apply_inverse(mesh_quat.unsqueeze(0).expand(self._num_points, -1), pts_w - mesh_pos)
                    if mesh_scale is not None:
                        s = mesh_scale.to(device=pts_l.device, dtype=pts_l.dtype).clamp(min=1.0e-12)
                        pts_l = pts_l / s.unsqueeze(0)
                    pts_l = pts_l.contiguous()
                    sdf_out = sdf_e[e].contiguous()

                    pts_wp = wp.from_torch(pts_l, dtype=wp.vec3)
                    sdf_wp = wp.from_torch(sdf_out, dtype=wp.float32)
                    wp.launch(
                        kernel=mesh_distance_kernel,
                        dim=self._num_points,
                        inputs=[
                            pts_wp,
                            wp_mesh.id,
                            tri_indices,
                            vertex_normals,
                            float(max_dist),
                            int(signed_mode),
                            int(smooth_normals),
                            sdf_wp,
                        ],
                        device=self._wp_device,
                    )
            else:
                box_pos_w = self._box_pos_w
                box_quat_w = self._box_quat_w
                box_half_extents = self._box_half_extents
                assert box_pos_w is not None and box_quat_w is not None and box_half_extents is not None

                box_pos = wp.vec3(float(box_pos_w[0].item()), float(box_pos_w[1].item()), float(box_pos_w[2].item()))
                box_quat = wp.vec4(
                    float(box_quat_w[0].item()),
                    float(box_quat_w[1].item()),
                    float(box_quat_w[2].item()),
                    float(box_quat_w[3].item()),
                )
                he = wp.vec3(
                    float(box_half_extents[0].item()),
                    float(box_half_extents[1].item()),
                    float(box_half_extents[2].item()),
                )

                for e in range(num_envs):
                    pts = points_w[e].contiguous()
                    sdf_out = sdf_e[e].contiguous()

                    pts_wp = wp.from_torch(pts, dtype=wp.vec3)
                    sdf_wp = wp.from_torch(sdf_out, dtype=wp.float32)

                    wp.launch(
                        kernel=box_sdf_kernel,
                        dim=self._num_points,
                        inputs=[pts_wp, box_pos, box_quat, he, sdf_wp],
                        device=self._wp_device,
                    )

            # Simple spring force.
            if use_mesh and not bool(getattr(self.cfg, "mesh_use_signed_distance", False)):
                # Unsigned distance: use a small shell thickness to emulate penetration.
                shell = float(getattr(self.cfg, "mesh_shell_thickness", 0.001))
                penetration = (shell - sdf_e).clamp_min(0.0)
            else:
                # Signed distance (or analytic box SDF): negative means inside.
                penetration = (-sdf_e).clamp_min(0.0)

                
            #mian fuction
            fn = (float(self.cfg.stiffness) * penetration).clamp(0.0, float(self.cfg.max_force))
            if self.cfg.normalize_forces:
                fn = fn / float(self.cfg.max_force)

            tactile = torch.cat((points_w, fn.unsqueeze(-1)), dim=-1)
            tactile_per_sensor.append(tactile)
            sdf_per_sensor.append(sdf_e)

        tactile_stack = torch.stack(tactile_per_sensor, dim=1)  # (E, S, P, 4)
        sdf_stack = torch.stack(sdf_per_sensor, dim=1)  # (E, S, P)

        self._data.tactile_points_w_per_sensor[env_ids] = tactile_stack
        self._data.tactile_points_w[env_ids] = tactile_stack.view(num_envs, self._num_sensors * self._num_points, 4)
        self._sdf_out[env_ids] = sdf_stack

        # Debug visualization: show taxels with fn > threshold for a chosen env.
        if self.cfg.debug_vis and self._debug_markers is not None:
            debug_env_id = int(getattr(self.cfg, "debug_vis_env_id", 0))
            if isinstance(env_ids, torch.Tensor):
                env_ids_list = env_ids.detach().cpu().tolist()
            else:
                env_ids_list = list(env_ids)
            if debug_env_id in env_ids_list:
                tactile_env = self._data.tactile_points_w_per_sensor[debug_env_id]  # (S, P, 4)
                pts = tactile_env[..., :3].reshape(-1, 3)
                fn = tactile_env[..., 3].reshape(-1)
                thr = float(getattr(self.cfg, "debug_vis_force_threshold", 1.0e-6))
                contact = fn > thr
                # 0 -> contact prototype, 1 -> no_contact prototype (hidden unless show_all enabled)
                proto = torch.where(
                    contact,
                    torch.zeros_like(fn, dtype=torch.int64),
                    torch.ones_like(fn, dtype=torch.int64),
                )
                self._debug_markers.visualize(translations=pts, marker_indices=proto)

                # Also visualize per-sensor body frames (XYZ axes) to inspect orientation.
                if self._debug_axes is not None:
                    trans_list = []
                    quat_list = []
                    # Visualize axes at the tactile patch frame (not the raw rigid-body origin),
                    # so users can validate/adjust patch_offset_{pos,quat} and normal_offset.
                    patch_pos_b = torch.tensor(
                        self._resolve_patch_offset_pos_list(),
                        device=self.device,
                        dtype=torch.float32,
                    )  # (S, 3)
                    patch_quat_b = torch.tensor(
                        self._resolve_patch_offset_quat_list(),
                        device=self.device,
                        dtype=torch.float32,
                    )  # (S, 4)

                    # normal_offset is applied along normal_axis in the patch-local frame when generating taxels.
                    normal_axis = int(self.cfg.normal_axis)
                    n_local = torch.zeros((self._num_sensors, 3), device=self.device, dtype=torch.float32)
                    n_local[:, normal_axis] = float(self.cfg.normal_offset)

                    for sensor_idx, s in enumerate(self._pose_sensors):
                        if not s.is_initialized:
                            continue
                        sd = s.data
                        if sd.pos_w is None or sd.quat_w is None:
                            continue

                        body_pos_w = sd.pos_w[debug_env_id, 0]
                        body_quat_w = sd.quat_w[debug_env_id, 0]

                        # patch orientation in world
                        patch_quat_w = quat_mul(body_quat_w.unsqueeze(0), patch_quat_b[sensor_idx].unsqueeze(0)).squeeze(0)
                        # patch origin in world: body origin + rotated patch offset + rotated normal offset
                        off_w = quat_apply(body_quat_w.unsqueeze(0), patch_pos_b[sensor_idx].unsqueeze(0)).squeeze(0)
                        n_off_w = quat_apply(patch_quat_w.unsqueeze(0), n_local[sensor_idx].unsqueeze(0)).squeeze(0)
                        patch_pos_w = body_pos_w + off_w + n_off_w

                        trans_list.append(patch_pos_w)
                        quat_list.append(patch_quat_w)
                    if trans_list:
                        trans = torch.stack(trans_list, dim=0)
                        quats = torch.stack(quat_list, dim=0)
                        idx = torch.zeros((trans.shape[0],), device=trans.device, dtype=torch.int64)
                        self._debug_axes.visualize(translations=trans, orientations=quats, marker_indices=idx)

    @staticmethod
    def _create_local_grid_points(
        *,
        num_rows: int,
        num_cols: int,
        point_distance: float,
        normal_axis: int,
        normal_offset: float,
        device: str,
    ) -> torch.Tensor:
        tangential_axes = [0, 1, 2]
        tangential_axes.remove(normal_axis)
        axis_u, axis_v = tangential_axes

        u = torch.linspace(
            -point_distance * (num_rows + 1) / 2.0,
            +point_distance * (num_rows + 1) / 2.0,
            steps=num_rows + 2,
            device=device,
            dtype=torch.float32,
        )[1:-1]
        v = torch.linspace(
            -point_distance * (num_cols + 1) / 2.0,
            +point_distance * (num_cols + 1) / 2.0,
            steps=num_cols + 2,
            device=device,
            dtype=torch.float32,
        )[1:-1]

        uu, vv = torch.meshgrid(u, v, indexing="ij")
        points = torch.zeros((num_rows * num_cols, 3), device=device, dtype=torch.float32)
        points[:, axis_u] = uu.reshape(-1)
        points[:, axis_v] = vv.reshape(-1)
        points[:, normal_axis] = float(normal_offset)
        return points

    def _resolve_patch_offset_pos_list(self) -> list[tuple[float, float, float]]:
        per = getattr(self.cfg, "patch_offset_pos_b_per_elastomer", None)
        if per is not None:
            if len(per) != self._num_sensors:
                raise ValueError(
                    "'patch_offset_pos_b_per_elastomer' must have the same length as 'elastomer_prim_paths'. "
                    f"Got {len(per)} vs {self._num_sensors}."
                )
            return list(per)
        base = getattr(self.cfg, "patch_offset_pos_b", (0.0, 0.0, 0.0))
        base_tuple: tuple[float, float, float] = (float(base[0]), float(base[1]), float(base[2]))
        return [base_tuple for _ in range(self._num_sensors)]

    def _resolve_patch_offset_quat_list(self) -> list[tuple[float, float, float, float]]:
        per = getattr(self.cfg, "patch_offset_quat_b_per_elastomer", None)
        if per is not None:
            if len(per) != self._num_sensors:
                raise ValueError(
                    "'patch_offset_quat_b_per_elastomer' must have the same length as 'elastomer_prim_paths'. "
                    f"Got {len(per)} vs {self._num_sensors}."
                )
            return list(per)
        base = getattr(self.cfg, "patch_offset_quat_b", (1.0, 0.0, 0.0, 0.0))
        base_tuple: tuple[float, float, float, float] = (
            float(base[0]),
            float(base[1]),
            float(base[2]),
            float(base[3]),
        )
        return [base_tuple for _ in range(self._num_sensors)]

    @staticmethod
    def _triangulate_usd_mesh(face_counts, face_indices):
        # Fan triangulation for polygon faces.
        tris = []
        idx = 0
        for n in face_counts:
            n = int(n)
            if n < 3:
                idx += n
                continue
            v0 = int(face_indices[idx])
            for i in range(1, n - 1):
                v1 = int(face_indices[idx + i])
                v2 = int(face_indices[idx + i + 1])
                tris.append((v0, v1, v2))
            idx += n
        return tris

    def _load_warp_mesh_from_usd(self, prim_path: str, device: str):
        import numpy as np
        from pxr import UsdGeom  # type: ignore[import-not-found]

        stage_prim = self.stage.GetPrimAtPath(prim_path)
        if stage_prim.IsValid() and stage_prim.GetTypeName() == "Mesh":
            mesh_prim = stage_prim
        else:
            mesh_prim = sim_utils.get_first_matching_child_prim(prim_path, lambda prim: prim.GetTypeName() == "Mesh")
        if mesh_prim is None or not mesh_prim.IsValid():
            raise RuntimeError(f"Invalid mesh prim path (no UsdGeom.Mesh found under): {prim_path}")

        mesh = UsdGeom.Mesh(mesh_prim)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float32)

        face_counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int32)
        face_indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
        if face_counts.size == 0 or face_indices.size == 0:
            raise RuntimeError(f"Mesh prim has no faces: {mesh.GetPath()}")

        if np.all(face_counts == 3) and (face_indices.size % 3 == 0):
            triangles = face_indices.reshape(-1, 3)
        else:
            triangles = np.asarray(self._triangulate_usd_mesh(face_counts, face_indices), dtype=np.int32)
            if triangles.size == 0:
                raise RuntimeError(f"Failed to triangulate mesh prim: {mesh.GetPath()}")

        # Compute vertex normals (area-weighted)
        v0 = points[triangles[:, 0]]
        v1 = points[triangles[:, 1]]
        v2 = points[triangles[:, 2]]
        face_normals = np.cross(v1 - v0, v2 - v0)
        vert_normals = np.zeros_like(points, dtype=np.float32)
        np.add.at(vert_normals, triangles[:, 0], face_normals)
        np.add.at(vert_normals, triangles[:, 1], face_normals)
        np.add.at(vert_normals, triangles[:, 2], face_normals)
        nrm = np.linalg.norm(vert_normals, axis=1, keepdims=True)
        vert_normals = np.divide(
            vert_normals,
            np.clip(nrm, 1.0e-12, None),
            out=np.zeros_like(vert_normals, dtype=np.float32),
            where=nrm > 0.0,
        ).astype(np.float32)

        wp_mesh = convert_to_warp_mesh(points, triangles, device=device)
        wp_tri_indices = wp.array(triangles.astype(np.int32).flatten(), dtype=wp.int32, device=device)
        wp_vertex_normals = wp.array(vert_normals.astype(np.float32), dtype=wp.vec3, device=device)
        return wp_mesh, wp_tri_indices, wp_vertex_normals
