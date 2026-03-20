"""Standalone gymnasium.Env for ALOHA bimanual robot with WarpSdf tactile sensors.

Accepts 16-dim joint position targets, returns tactile force grids,
joint states, object poses, and RGB renders.
"""

from __future__ import annotations

import os
import gymnasium
import numpy as np
import torch
from utils.utlis_misc import _parse_elastomer_origins, _infer_arm, _infer_finger, _sensor_slot
from utils.utils import _xyzw_to_wxyz, _look_at_quat, _resolve_joint_ids
from aloha_tactile_cfg import AlohaTactileEnvCfg, DATASET_JOINT_ORDER, TrackInfo




class AlohaTactileEnv(gymnasium.Env):
    """ALOHA bimanual tactile environment.

    Observations:
        tactile:     (4, num_rows, num_cols) force grids
        joint_pos:   (16,) joint positions in dataset order
        joint_vel:   (16,) joint velocities in dataset order
        plug_pose:   (7,) pos + quat_wxyz (zeros if disabled)
        socket_pose: (7,) pos + quat_wxyz (zeros if disabled)
        rgb:         (H, W, 3) uint8 (if enable_camera)

    Actions:
        Box(16,) raw joint position targets.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, cfg: AlohaTactileEnvCfg, simulation_app=None):
        super().__init__()
        self._cfg = cfg
        self._simulation_app = simulation_app
        self._step_count = 0

        # Deferred imports (require SimulationApp)
        import isaacsim.core.utils.prims as prim_utils
        from isaacsim.core.api.simulation_context import SimulationContext
        from isaacsim.core.utils.extensions import enable_extension

        import isaaclab.sim as sim_utils
        import isaaclab.utils.math as math_utils
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import Articulation, RigidObject
        from isaaclab.assets.articulation import ArticulationCfg
        from isaaclab.assets.rigid_object import RigidObjectCfg
        from isaaclab.sensors.warp_sdf_tactile import WarpSdfTactileSensor, WarpSdfTactileSensorCfg
        from isaaclab.sim.converters import UrdfConverterCfg
        from isaaclab.sim.schemas import activate_contact_sensors

        self._prim_utils = prim_utils
        self._sim_utils = sim_utils
        self._math_utils = math_utils

        enable_extension("isaacsim.asset.importer.urdf")

        self._init_sim(SimulationContext, cfg)
        self._spawn_basic_world(sim_utils)

        # Camera
        self._setup_camera(cfg, sim_utils)

        # Parse URDF elastomer origins
        urdf_path = os.path.expanduser(cfg.urdf_path)
        self._urdf_origins = _parse_elastomer_origins(urdf_path)

        self._plug_obj, self._socket_obj = self._spawn_plug_socket(cfg, sim_utils, RigidObject, RigidObjectCfg)
        
        # Spawn robot
        out_dir = cfg.usd_output_dir or os.path.join(os.path.dirname(__file__), "output", "aloha_urdf")
        os.makedirs(out_dir, exist_ok=True)

        self._robot = self._spawn_robot(
            cfg,
            urdf_path,
            sim_utils,
            Articulation,
            ArticulationCfg,
            ImplicitActuatorCfg,
            UrdfConverterCfg,
        )
        activate_contact_sensors(cfg.robot_prim_path, threshold=0.0)

        # Find and sort the 4 elastomer links
        from pxr import PhysxSchema, UsdPhysics
        self._UsdPhysics = UsdPhysics

        elastomers = self._find_elastomer_links(cfg, sim_utils, UsdPhysics, PhysxSchema)
        selected = self._sort_elastomer_links(elastomers)
        self._selected_links = selected

        # Resolve per-sensor target mesh prims
        target_root_paths, target_query_paths = self._build_target_query_paths(
            cfg, selected, prim_utils, sim_utils
        )
        self._per_sensor_target_query_paths = target_query_paths

        for i, (link, root, query) in enumerate(zip(selected, target_root_paths, target_query_paths)):
            print(
                f"  [{i}] slot={_sensor_slot(link)} arm={_infer_arm(link) or '?':>5s}"
                f" elastomer={link} -> query={query}",
                flush=True,
            )


        # Compute patch offsets per elastomer and create tactile sensors
        self._tactile_sensors, self._sensor_slot_order = self._create_tactile_sensors(
            cfg,
            selected,
            target_query_paths,
            WarpSdfTactileSensor,
            WarpSdfTactileSensorCfg,
            math_utils,
        )

        # Reset simulation (triggers sensor PLAY callbacks)
        self._post_spawn_init(cfg, sim_utils, target_query_paths)
        self._build_spaces(cfg)

    # -------------------------------------------------------------------
    # Gym interface
    # -------------------------------------------------------------------

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).reshape(16)
        action_t = torch.tensor(action, dtype=torch.float32, device=self._device)

        self._robot.set_joint_position_target(action_t, joint_ids=self._dataset_joint_ids)
        self._robot.write_data_to_sim()

        render = not self._cfg.headless or self._camera is not None
        self._sim.step(render=render)

        dt = self._cfg.physics_dt
        self._robot.update(dt)
        if self._plug_obj:
            self._plug_obj.update(dt)
        if self._socket_obj:
            self._socket_obj.update(dt)

        self._update_target_poses()
        for sensor in self._tactile_sensors:
            sensor.update(dt=dt)
        if self._camera:
            self._camera.update(dt=dt)

        obs = self._get_obs()

        if self._render_output_dir and self._camera:
            self._save_render(obs.get("rgb"), self._step_count)

        self._step_count += 1
        return obs, 0.0, False, False, {}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self._robot.write_joint_state_to_sim(
            self._robot.data.default_joint_pos.clone(),
            self._robot.data.default_joint_vel.clone(),
        )
        self._robot.reset()

        for obj in (self._plug_obj, self._socket_obj):
            if obj:
                obj.write_root_state_to_sim(obj.data.default_root_state)
                obj.reset()

        render = not self._cfg.headless or self._camera is not None
        self._sim.step(render=render)
        dt = self._cfg.physics_dt
        self._robot.update(dt)
        for obj in (self._plug_obj, self._socket_obj):
            if obj:
                obj.update(dt)

        for sensor in self._tactile_sensors:
            sensor.reset()
        if self._camera:
            self._camera.reset()
            self._camera.update(dt=dt)

        self._step_count = 0
        return self._get_obs(), {}

    def close(self):
        self._tactile_sensors.clear()

    # -------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------

    def _get_obs(self) -> dict:
        cfg = self._cfg

        tactile = np.zeros((4, cfg.num_rows, cfg.num_cols), dtype=np.float32)
        for i, sensor in enumerate(self._tactile_sensors):
            data = sensor.data.tactile_points_w
            if data is not None:
                forces = data[0, :, 3].detach().cpu().numpy().astype(np.float32)
                tactile[self._sensor_slot_order[i]] = forces.reshape(cfg.num_rows, cfg.num_cols)

        ids = self._dataset_joint_ids
        joint_pos = self._robot.data.joint_pos[0, ids].detach().cpu().numpy().astype(np.float32)
        joint_vel = self._robot.data.joint_vel[0, ids].detach().cpu().numpy().astype(np.float32)

        obs = {
            "tactile": tactile,
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "plug_pose": _obj_pose_numpy(self._plug_obj),
            "socket_pose": _obj_pose_numpy(self._socket_obj),
        }

        if self._camera:
            try:
                rgb = self._camera.data.output["rgb"][0, :, :, :3].detach().cpu().numpy().astype(np.uint8)
                obs["rgb"] = rgb
            except Exception:
                obs["rgb"] = np.zeros((cfg.camera_height, cfg.camera_width, 3), dtype=np.uint8)

        return obs

    def _update_target_poses(self):
        for sensor, tgt_prim in zip(self._tactile_sensors, self._per_sensor_target_prims):
            if not tgt_prim or not tgt_prim.IsValid():
                continue

            path = tgt_prim.GetPath().pathString
            info = self._dynamic_track_map.get(path)

            if info is not None:
                try:
                    pos, quat = self._tracked_target_pose(info)
                except Exception:
                    pos, quat = self._sim_utils.resolve_prim_pose(tgt_prim)
            else:
                pos, quat = self._sim_utils.resolve_prim_pose(tgt_prim)

            sensor.set_target_pose(pos, quat)

    def _setup_dynamic_tracking(self):
        """Build map: target mesh prim path -> TrackInfo (best-effort)."""
        math_utils = self._math_utils
        sim_utils = self._sim_utils
        UsdPhysics = self._UsdPhysics

        # Try to import RigidPrim (API name varies)
        RigidPrim = None
        for mod in ("omni.isaac.core.prims", "isaacsim.core.prims"):
            try:
                RigidPrim = __import__(mod, fromlist=["RigidPrim"]).RigidPrim
                break
            except ImportError:
                continue

        self._dynamic_track_map = {}
        if RigidPrim is None:
            return

        def make_rigid_prim(path: str):
            """Try common ctor signatures; return constructed RigidPrim or raise last error."""
            candidates = (
                ((), {}),                         # RigidPrim(path) handled outside
                ((), {"prim_path": path}),
                ((), {"path": path}),             # occasionally seen, harmless if wrong
                ((), {"name": path.replace("/", "_")}),
            )

            # First: many versions accept positional path
            try:
                return RigidPrim(path)
            except TypeError:
                pass

            last_err = None
            for _, kwargs in candidates:
                try:
                    # Some candidates above don't include prim_path; add it when needed
                    if "prim_path" not in kwargs and "path" not in kwargs:
                        continue
                    return RigidPrim(**kwargs)
                except TypeError as e:
                    last_err = e
                    continue

            # Fallback (most common keyword)
            if last_err is not None:
                return RigidPrim(prim_path=path)
            return RigidPrim(prim_path=path)

        def to_t(x):
            return torch.tensor(x, device=self._device, dtype=torch.float32)

        for query_path in self._per_sensor_target_query_paths:
            if not query_path:
                continue

            prim = self._stage.GetPrimAtPath(query_path)
            if not prim.IsValid():
                continue

            # Find parent rigid body
            curr = prim
            rb_prim = None
            while curr.IsValid() and not curr.IsPseudoRoot():
                if curr.HasAPI(UsdPhysics.RigidBodyAPI) or curr.HasAPI(UsdPhysics.MassAPI):
                    rb_prim = curr
                    break
                curr = curr.GetParent()
            if rb_prim is None:
                continue

            rb_path = rb_prim.GetPath().pathString

            # Single best-effort try/except per query_path
            try:
                rp = make_rigid_prim(rb_path)
                if hasattr(rp, "initialize"):
                    rp.initialize()

                p_m, q_m = sim_utils.resolve_prim_pose(prim)
                p_b, q_b = sim_utils.resolve_prim_pose(rb_prim)

                q_b_inv = math_utils.quat_inv(to_t(q_b))
                p_rel = math_utils.quat_apply(q_b_inv, to_t(p_m) - to_t(p_b))
                q_rel = math_utils.quat_mul(q_b_inv, to_t(q_m))

                self._dynamic_track_map[query_path] = TrackInfo(
                    rp=rp, p_rel=p_rel, q_rel=q_rel, rb_path=rb_path
                )

            except Exception as e:
                print(f"[WARN] Dynamic tracking init failed for {rb_path}: {e}", flush=True)


    def _tracked_target_pose(self, info: TrackInfo):
        """Return (pos, quat) numpy for the tracked target pose. Raises on failure."""
        pos_b, quat_b = _rigid_prim_world_pose(info.rp)

        pos_b_t = torch.tensor(pos_b, device=self._device, dtype=torch.float32)
        quat_b_t = torch.tensor(quat_b, device=self._device, dtype=torch.float32)

        pos_t = pos_b_t + self._math_utils.quat_apply(quat_b_t, info.p_rel)
        quat_t = self._math_utils.quat_mul(quat_b_t, info.q_rel)

        return pos_t.detach().cpu().numpy(), quat_t.detach().cpu().numpy()


    def _save_render(self, rgb: np.ndarray | None, step: int):
        if rgb is None or not self._render_output_dir:
            return
        try:
            from PIL import Image
            Image.fromarray(rgb).save(os.path.join(self._render_output_dir, f"frame_{step:06d}.png"))
        except ImportError:
            np.save(os.path.join(self._render_output_dir, f"frame_{step:06d}.npy"), rgb)

    def _init_sim(self, SimulationContext, cfg: AlohaTactileEnvCfg) -> None:
        """Create simulation context and cache device."""
        self._sim = SimulationContext(
            physics_dt=cfg.physics_dt,
            rendering_dt=cfg.physics_dt,
            backend="torch",
            device=cfg.device,
        )
        self._device = cfg.device


    def _spawn_basic_world(self, sim_utils) -> None:
        """Spawn ground plane and dome light."""
        # Ground plane
        sim_utils.spawn_mesh_cuboid(
            prim_path="/World/defaultGroundPlane",
            cfg=sim_utils.MeshCuboidCfg(
                size=(10.0, 10.0, 0.1),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=0.004, rest_offset=0.0
                ),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True, disable_gravity=True
                ),
            ),
            translation=(0.0, 0.0, -0.05),
            orientation=(1.0, 0.0, 0.0, 0.0),
        )

        # Dome light
        sim_utils.spawn_light(
            prim_path="/World/Light/DomeLight",
            cfg=sim_utils.DomeLightCfg(intensity=2000),
            translation=(-4.5, 3.5, 10.0),
        )

    def _setup_camera(self, cfg: AlohaTactileEnvCfg, sim_utils):
        """Create camera (optional) and set render output dir (optional)."""
        self._camera = None
        self._render_output_dir = None

        if not cfg.enable_camera:
            return

        # Deferred (camera modules rely on Isaac)
        from isaacsim.core.utils.viewports import set_camera_view
        from isaaclab.sensors.camera import Camera, CameraCfg

        set_camera_view(list(cfg.camera_eye), list(cfg.camera_target))
        cam_rot = _look_at_quat(cfg.camera_eye, cfg.camera_target)

        self._camera = Camera(CameraCfg(
            prim_path=cfg.camera_prim_path,
            update_period=0.0,
            height=cfg.camera_height,
            width=cfg.camera_width,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 1.0e5),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=cfg.camera_eye,
                rot=cam_rot,
                convention="world",
            ),
        ))

        if cfg.save_renders:
            self._render_output_dir = cfg.render_output_dir or os.path.join(
                os.path.dirname(__file__), "output", "renders"
            )
            os.makedirs(self._render_output_dir, exist_ok=True)


    def _spawn_plug_socket(self, cfg: AlohaTactileEnvCfg, sim_utils, RigidObject, RigidObjectCfg):
        """Optionally spawn plug and/or socket RigidObjects. Returns (plug, socket)."""
        plug_obj = None
        socket_obj = None

        # Always ensure output dir exists (used by USD conversion)
        objs_out_dir = os.path.join(os.path.dirname(__file__), "output", "automate_scaled_urdf")
        os.makedirs(objs_out_dir, exist_ok=True)

        if not (cfg.enable_plug or cfg.enable_socket):
            return plug_obj, socket_obj

        automate_dir = os.path.join(os.path.expanduser(cfg.asset_root), "automate_scaled", "urdf")
        plug_urdf = os.path.join(automate_dir, f"{cfg.automate_asset_id}_plug.urdf")
        socket_urdf = os.path.join(automate_dir, f"{cfg.automate_asset_id}_socket.urdf")

        def spawn_one(urdf_path: str, pose, prim_path: str, scale: float, fix_base: bool, collider_type: str):
            if not os.path.isfile(urdf_path):
                return None
            pos = tuple(float(v) for v in pose[:3])
            rot = _xyzw_to_wxyz(pose[3:7])
            return RigidObject(RigidObjectCfg(
                prim_path=prim_path,
                spawn=sim_utils.UrdfFileCfg(
                    asset_path=urdf_path,
                    scale=(scale,) * 3 if scale != 1.0 else None,
                    fix_base=fix_base,
                    joint_drive=None,
                    link_density=1000.0,
                    usd_dir=objs_out_dir,
                    force_usd_conversion=cfg.force_objects_urdf_conversion,
                    collider_type=collider_type,
                    activate_contact_sensors=False,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=rot),
            ))

        if cfg.enable_plug:
            plug_obj = spawn_one(
                plug_urdf, cfg.plug_default_pose, "/World/Plug",
                cfg.plug_scale, cfg.plug_fix_base, cfg.plug_collider_type,
            )

        if cfg.enable_socket:
            socket_obj = spawn_one(
                socket_urdf, cfg.socket_default_pose, "/World/Socket",
                cfg.socket_scale, cfg.socket_fix_base, cfg.socket_collider_type,
            )

        return plug_obj, socket_obj


    def _spawn_robot(
        self,
        cfg: AlohaTactileEnvCfg,
        urdf_path: str,
        sim_utils,
        Articulation,
        ArticulationCfg,
        ImplicitActuatorCfg,
        UrdfConverterCfg,
    ):
        """Spawn the ALOHA articulation from URDF and return the Articulation."""
        out_dir = cfg.usd_output_dir or os.path.join(os.path.dirname(__file__), "output", "aloha_urdf")
        os.makedirs(out_dir, exist_ok=True)

        robot = Articulation(ArticulationCfg(
            prim_path=cfg.robot_prim_path,
            spawn=sim_utils.UrdfFileCfg(
                asset_path=urdf_path,
                fix_base=cfg.fix_base,
                merge_fixed_joints=cfg.merge_fixed_joints,
                joint_drive=UrdfConverterCfg.JointDriveCfg(
                    gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                        stiffness=cfg.urdf_drive_stiffness,
                        damping=cfg.urdf_drive_damping,
                    )
                ),
                usd_dir=out_dir,
                force_usd_conversion=cfg.force_urdf_conversion,
                activate_contact_sensors=True,
            ),
            init_state=ArticulationCfg.InitialStateCfg(pos=(0, 0, 0), rot=(1, 0, 0, 0)),
            actuators={
                "all": ImplicitActuatorCfg(
                    joint_names_expr=[".*"],
                    stiffness=cfg.urdf_drive_stiffness,
                    damping=cfg.urdf_drive_damping,
                )
            },
        ))
        return robot


    def _find_elastomer_links(self, cfg: AlohaTactileEnvCfg, sim_utils, UsdPhysics, PhysxSchema) -> list[str]:
        """Return elastomer link prim paths (strings)."""
        bodies = sim_utils.get_all_matching_child_prims(
            cfg.robot_prim_path,
            predicate=lambda p: p.HasAPI(UsdPhysics.RigidBodyAPI) and p.HasAPI(PhysxSchema.PhysxContactReportAPI),
            traverse_instance_prims=False,
        )
        elastomers = sorted(
            [p.GetPath().pathString for p in bodies if "elastomer" in p.GetPath().pathString.lower()]
        )[:4]
        if not elastomers:
            raise RuntimeError("No elastomer links found on robot.")
        return elastomers

    def _sort_elastomer_links(self, elastomers: list[str]) -> list[str]:
        """Sort elastomer paths into deterministic arm/finger order."""
        def sort_key(path: str):
            arm = _infer_arm(path)
            arm_order = 0 if arm == "left" else 1

            finger = _infer_finger(path)
            finger_order = 0 if finger == "left_finger" else 1

            return (arm_order, finger_order, path)

        return sorted(elastomers, key=sort_key)
    
    def _build_target_query_paths(
        self,
        cfg: AlohaTactileEnvCfg,
        selected_links: list[str],
        prim_utils,
        sim_utils,
    ) -> tuple[list[str], list[str]]:
        """Return (target_root_paths, target_query_paths) per selected elastomer link."""
        left_target = cfg.left_arm_target_mesh_prim if self._socket_obj else None
        right_target = cfg.right_arm_target_mesh_prim if self._plug_obj else None

        target_root_paths: list[str] = []
        for link_path in selected_links:
            arm = _infer_arm(link_path)
            if arm == "left" and left_target:
                target_root_paths.append(left_target)
            elif arm == "right" and right_target:
                target_root_paths.append(right_target)
            else:
                target_root_paths.append(cfg.robot_prim_path)

        target_query_paths: list[str] = []
        for root in target_root_paths:
            try:
                qp, _ = _resolve_mesh_prim(root, prim_utils=prim_utils, sim_utils=sim_utils)
            except RuntimeError:
                qp = root
            target_query_paths.append(qp)

        return target_root_paths, target_query_paths

    def _compute_patch_transform(self, link_path: str, cfg: AlohaTactileEnvCfg, math_utils):
        """Compute patch offset pos/quat in body frame for a given elastomer link."""
        lp = link_path.lower()
        if "_left_finger_link" in lp:
            side = "left"
        elif "_right_finger_link" in lp:
            side = "right"
        else:
            return cfg.patch_offset_pos, cfg.patch_offset_quat

        base_xyz, base_rpy = self._urdf_origins[side]

        base_xyz_t = torch.tensor(base_xyz, dtype=torch.float32).unsqueeze(0)
        base_rpy_t = torch.tensor(base_rpy, dtype=torch.float32).unsqueeze(0)
        user_pos_t = torch.tensor(cfg.patch_offset_pos, dtype=torch.float32).unsqueeze(0)
        user_quat_t = torch.tensor(cfg.patch_offset_quat, dtype=torch.float32).unsqueeze(0)

        q_be = math_utils.quat_from_euler_xyz(base_rpy_t[:, 0], base_rpy_t[:, 1], base_rpy_t[:, 2])
        pos_bp = (base_xyz_t + math_utils.quat_apply(q_be, user_pos_t)).squeeze(0)
        quat_bp = math_utils.quat_mul(q_be, user_quat_t).squeeze(0)

        return tuple(float(v) for v in pos_bp), tuple(float(v) for v in quat_bp)

    def _create_tactile_sensors(
        self,
        cfg: AlohaTactileEnvCfg,
        selected_links: list[str],
        target_query_paths: list[str],
        WarpSdfTactileSensor,
        WarpSdfTactileSensorCfg,
        math_utils,
    ):
        """Create tactile sensors and return (sensors, slot_order)."""
        sensors: list = []
        slot_order: list[int] = []

        for i, link_path in enumerate(selected_links):
            patch_pos, patch_quat = self._compute_patch_transform(link_path, cfg, math_utils)

            sensor_cfg = WarpSdfTactileSensorCfg(
                prim_path=cfg.robot_prim_path,
                elastomer_prim_paths=[link_path],
                num_rows=cfg.num_rows,
                num_cols=cfg.num_cols,
                point_distance=cfg.point_distance,
                normal_axis=cfg.normal_axis,
                normal_offset=cfg.normal_offset,
                patch_offset_pos_b=patch_pos,
                patch_offset_quat_b=patch_quat,
                target_mesh_prim_path=target_query_paths[i],
                mesh_max_dist=cfg.mesh_max_dist,
                mesh_use_signed_distance=cfg.mesh_signed,
                mesh_signed_distance_method=cfg.mesh_signed_distance_method,
                mesh_smooth_normals=True,
                mesh_shell_thickness=cfg.mesh_shell_thickness,
                stiffness=cfg.stiffness,
                max_force=cfg.max_force,
                normalize_forces=True,
                debug_vis=cfg.debug_vis,
            )
            sensors.append(WarpSdfTactileSensor(sensor_cfg))

            slot = _sensor_slot(link_path)
            slot_order.append(slot if slot is not None else i)

        return sensors, slot_order

    def _post_spawn_init(self, cfg: AlohaTactileEnvCfg, sim_utils, target_query_paths: list[str]) -> None:
        """Final initialization after spawning robot/objects/sensors."""
        # Reset simulation (triggers sensor PLAY callbacks)
        self._sim.reset()
        dt = cfg.physics_dt

        # First update pass so buffers are valid
        self._robot.update(dt)
        if self._plug_obj:
            self._plug_obj.update(dt)
        if self._socket_obj:
            self._socket_obj.update(dt)

        # Set up dynamic tracking
        self._stage = sim_utils.get_current_stage()
        self._per_sensor_target_prims = [self._stage.GetPrimAtPath(p) for p in target_query_paths]
        self._setup_dynamic_tracking()

        # Resolve joint mapping (dataset order)
        self._dataset_joint_ids = _resolve_joint_ids(self._robot, DATASET_JOINT_ORDER)

        # Logging
        print(
            f"[INFO] {len(self._dataset_joint_ids)} joints mapped, "
            f"{len(self._tactile_sensors)} tactile sensors",
            flush=True,
        )
        if self._camera:
            print(f"[INFO] Camera: {cfg.camera_width}x{cfg.camera_height}", flush=True)

    def _build_spaces(self, cfg: AlohaTactileEnvCfg) -> None:
        obs_spaces = {
            "tactile": gymnasium.spaces.Box(
                -np.inf, np.inf, shape=(4, cfg.num_rows, cfg.num_cols), dtype=np.float32
            ),
            "joint_pos": gymnasium.spaces.Box(-np.inf, np.inf, shape=(16,), dtype=np.float32),
            "joint_vel": gymnasium.spaces.Box(-np.inf, np.inf, shape=(16,), dtype=np.float32),
            "plug_pose": gymnasium.spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
            "socket_pose": gymnasium.spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
        }
        if self._camera:
            obs_spaces["rgb"] = gymnasium.spaces.Box(
                0, 255, shape=(cfg.camera_height, cfg.camera_width, 3), dtype=np.uint8
            )

        self.observation_space = gymnasium.spaces.Dict(obs_spaces)
        self.action_space = gymnasium.spaces.Box(-np.inf, np.inf, shape=(16,), dtype=np.float32)


def _resolve_mesh_prim(root_path, *, prim_utils, sim_utils):
    """Find the first Mesh prim under *root_path* for SDF queries."""
    from pxr import UsdGeom

    root_prim = prim_utils.get_prim_at_path(str(root_path))
    if not root_prim or not root_prim.IsValid():
        raise RuntimeError(f"Invalid target mesh prim: {root_path}")

    query_path = str(root_path)
    if not root_prim.IsA(UsdGeom.Mesh):
        children = sim_utils.get_all_matching_child_prims(
            query_path, predicate=lambda p: p.IsA(UsdGeom.Mesh),
            traverse_instance_prims=True,
        )
        if children:
            query_path = children[0].GetPath().pathString

    query_prim = prim_utils.get_prim_at_path(query_path)
    if not query_prim or not query_prim.IsValid():
        raise RuntimeError(f"No Mesh prim found under: {root_path}")
    return query_path, query_prim


def _to_numpy_1d(x, expected):
    x = to_numpy(x, shape=(-1,))
    if x.size != expected:
        raise ValueError(...)
    return x

def to_numpy(x, *, dtype=None, shape=None):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    if dtype is not None:
        x = x.astype(dtype, copy=False)
    if shape is not None:
        x = x.reshape(shape)
    return x

def _obj_pose_numpy(obj) -> np.ndarray:
    pose = np.zeros(7, dtype=np.float32)
    if obj is None:
        return pose
    # these should exist for IsaacLab RigidObject; if not, pose remains zeros
    root_pos = getattr(obj.data, "root_pos_w", None)
    root_quat = getattr(obj.data, "root_quat_w", None)
    if root_pos is None or root_quat is None:
        return pose
    pose[:3] = to_numpy(root_pos[0], dtype=np.float32, shape=(3,))
    pose[3:] = to_numpy(root_quat[0], dtype=np.float32, shape=(4,))
    return pose


def _rigid_prim_world_pose(rp):
    """Get (pos, quat) as numpy from a RigidPrim (handles API variations)."""
    fn = getattr(rp, "get_world_pose", None) or getattr(rp, "get_world_poses", None)
    if fn is None:
        raise AttributeError("RigidPrim has neither get_world_pose nor get_world_poses")
    pos, quat = fn()
    return _to_numpy_1d(pos, 3), _to_numpy_1d(quat, 4)