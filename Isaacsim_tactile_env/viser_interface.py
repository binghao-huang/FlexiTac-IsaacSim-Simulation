from __future__ import annotations

import argparse
import os
import sys
import threading
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# --- SimulationApp MUST be created before any omni/isaaclab imports ---
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="ALOHA bimanual drag control and visualization for current aloha_tactile_env")
parser.add_argument("--port", type=int, default=8080, help="Viser server port")
parser.add_argument("--save_dir", type=str, default="", help="Directory for saved trajectory .npz files")
parser.add_argument("--slow_interval", type=int, default=8, help="UI refresh interval for expensive updates")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# --- Now safe to import everything else ---
sys.path.insert(0, str(Path(__file__).parent))
from aloha_ik_controller import AlohaArmIKController
from aloha_tactile_env import AlohaTactileEnv, AlohaTactileEnvCfg


def patch_env_spawn_objects_for_socket_articulation_fix():
    """Spawn plug/socket normally, but disable articulation roots on converted object USDs when possible."""

    def _spawn_plug_socket_fixed(self, cfg, sim_utils, RigidObject, RigidObjectCfg):
        from utils.utils import _xyzw_to_wxyz

        plug_obj = None
        socket_obj = None

        objs_out_dir = os.path.join(os.path.dirname(__file__), "output", "automate_scaled_urdf")
        os.makedirs(objs_out_dir, exist_ok=True)

        if not (getattr(cfg, "enable_plug", False) or getattr(cfg, "enable_socket", False)):
            return plug_obj, socket_obj

        automate_dir = os.path.join(os.path.expanduser(cfg.asset_root), "automate_scaled", "urdf")
        plug_urdf = os.path.join(automate_dir, f"{cfg.automate_asset_id}_plug.urdf")
        socket_urdf = os.path.join(automate_dir, f"{cfg.automate_asset_id}_socket.urdf")

        def _make_spawn_cfg(urdf_path: str, fix_base: bool, collider_type: str):
            spawn_kwargs = dict(
                asset_path=urdf_path,
                scale=None,
                fix_base=fix_base,
                joint_drive=None,
                link_density=1000.0,
                usd_dir=objs_out_dir,
                force_usd_conversion=getattr(cfg, "force_objects_urdf_conversion", False),
                collider_type=collider_type,
                activate_contact_sensors=False,
            )
            spawn_cfg = sim_utils.UrdfFileCfg(**spawn_kwargs)

            try:
                art_root_cfg = None
                try:
                    from isaaclab.sim.schemas import ArticulationRootPropertiesCfg  # type: ignore
                    art_root_cfg = ArticulationRootPropertiesCfg(articulation_enabled=False)
                except Exception:
                    try:
                        from isaaclab.sim.schemas.schemas_cfg import ArticulationRootPropertiesCfg  # type: ignore
                        art_root_cfg = ArticulationRootPropertiesCfg(articulation_enabled=False)
                    except Exception:
                        art_root_cfg = None

                if art_root_cfg is not None:
                    if hasattr(spawn_cfg, "articulation_props"):
                        spawn_cfg.articulation_props = art_root_cfg
                    elif hasattr(spawn_cfg, "articulation_root_props"):
                        spawn_cfg.articulation_root_props = art_root_cfg
                    elif hasattr(spawn_cfg, "usd_config") and hasattr(spawn_cfg.usd_config, "articulation_props"):
                        spawn_cfg.usd_config.articulation_props = art_root_cfg
            except Exception as e:
                print(f"[WARN] Could not attach articulation-root fix for {os.path.basename(urdf_path)}: {e}", flush=True)

            return spawn_cfg

        def spawn_one(urdf_path: str, pose, prim_path: str, fix_base: bool, collider_type: str):
            if not os.path.isfile(urdf_path):
                return None
            pos = tuple(float(v) for v in pose[:3])
            rot = _xyzw_to_wxyz(pose[3:7])
            return RigidObject(RigidObjectCfg(
                prim_path=prim_path,
                spawn=_make_spawn_cfg(urdf_path, fix_base, collider_type),
                init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=rot),
            ))

        if getattr(cfg, "enable_plug", False):
            plug_obj = spawn_one(
                plug_urdf, cfg.plug_default_pose, "/World/Plug",
                getattr(cfg, "plug_fix_base", False), getattr(cfg, "plug_collider_type", "convex_decomposition"),
            )

        if getattr(cfg, "enable_socket", False):
            socket_obj = spawn_one(
                socket_urdf, cfg.socket_default_pose, "/World/Socket",
                getattr(cfg, "socket_fix_base", False), getattr(cfg, "socket_collider_type", "convex_decomposition"),
            )

        return plug_obj, socket_obj

    AlohaTactileEnv._spawn_plug_socket = _spawn_plug_socket_fixed


# ---------------------------------------------------------------------------
# Tactile heatmap rendering
# ---------------------------------------------------------------------------

def _jet_colormap(v: np.ndarray) -> np.ndarray:
    r = np.clip(1.5 - np.abs(4.0 * v - 3.0), 0, 1)
    g = np.clip(1.5 - np.abs(4.0 * v - 2.0), 0, 1)
    b = np.clip(1.5 - np.abs(4.0 * v - 1.0), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def tactile_to_rgb(grid: np.ndarray, scale: int = 8) -> np.ndarray:
    from PIL import Image

    vmax = float(grid.max())
    normed = grid / (vmax + 1e-8) if vmax > 0 else np.zeros_like(grid)
    rgb = _jet_colormap(normed)
    img = Image.fromarray(rgb)
    img = img.resize((grid.shape[1] * scale, grid.shape[0] * scale), Image.NEAREST)
    return np.array(img)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_mesh_path_from_urdf(urdf_path: str):
    urdf_path = os.path.expanduser(urdf_path)
    if not os.path.isfile(urdf_path):
        return None
    tree = ET.parse(urdf_path)
    for visual in tree.getroot().iter("visual"):
        geom = visual.find("geometry")
        if geom is None:
            continue
        mesh_el = geom.find("mesh")
        if mesh_el is None:
            continue
        filename = mesh_el.get("filename", "")
        scale_str = mesh_el.get("scale", "1 1 1")
        scale = tuple(float(v) for v in scale_str.split())
        if len(scale) != 3:
            scale = (1.0, 1.0, 1.0)
        mesh_abs = os.path.normpath(os.path.join(os.path.dirname(urdf_path), filename))
        if os.path.isfile(mesh_abs):
            return mesh_abs, scale
    return None


def _quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float64)


# ---------------------------------------------------------------------------
# Thread-safe shared state
# ---------------------------------------------------------------------------

class SharedState:
    def __init__(self):
        self._lock = threading.Lock()
        self._targets = {
            "left": {
                "pos": [0.0, -0.167, 0.074],
                "quat": [1.0, 0.0, 0.0, 0.0],
                "gripper": 1.0,
                "dragging": False,
            },
            "right": {
                "pos": [0.0, 0.167, 0.074],
                "quat": [1.0, 0.0, 0.0, 0.0],
                "gripper": 1.0,
                "dragging": False,
            },
        }
        self._reset_requested = False
        self._recording = False

    def set_target(self, arm: str, x, y, z):
        with self._lock:
            self._targets[arm]["pos"] = [float(x), float(y), float(z)]

    def set_target_quat(self, arm: str, w, x, y, z):
        with self._lock:
            self._targets[arm]["quat"] = [float(w), float(x), float(y), float(z)]

    def get_target(self, arm: str):
        with self._lock:
            return list(self._targets[arm]["pos"])

    def get_target_quat(self, arm: str):
        with self._lock:
            return list(self._targets[arm]["quat"])

    def set_gripper(self, arm: str, g):
        with self._lock:
            self._targets[arm]["gripper"] = float(g)

    def get_gripper(self, arm: str):
        with self._lock:
            return float(self._targets[arm]["gripper"])

    def request_reset(self):
        with self._lock:
            self._reset_requested = True

    def consume_reset(self):
        with self._lock:
            if self._reset_requested:
                self._reset_requested = False
                return True
            return False

    def set_dragging(self, arm: str, v):
        with self._lock:
            self._targets[arm]["dragging"] = bool(v)

    def is_dragging(self, arm: str):
        with self._lock:
            return bool(self._targets[arm]["dragging"])

    def set_recording(self, v):
        with self._lock:
            self._recording = bool(v)

    def is_recording(self):
        with self._lock:
            return self._recording


shared = SharedState()


# ---------------------------------------------------------------------------
# Joint name mapping
# ---------------------------------------------------------------------------

DATASET_JOINT_ORDER = [
    "left/waist", "left/shoulder", "left/elbow", "left/forearm_roll",
    "left/wrist_angle", "left/wrist_rotate", "left/left_finger", "left/right_finger",
    "right/waist", "right/shoulder", "right/elbow", "right/forearm_roll",
    "right/wrist_angle", "right/wrist_rotate", "right/left_finger", "right/right_finger",
]


def build_joint_name_map(urdf_joint_names):
    mapping = []
    dataset_lower = [n.lower() for n in DATASET_JOINT_ORDER]
    for urdf_name in urdf_joint_names:
        name = urdf_name.lower()
        found = None
        if name in dataset_lower:
            found = dataset_lower.index(name)
        else:
            name_underscore = name.replace("/", "_")
            for j, dt in enumerate(dataset_lower):
                if name_underscore == dt.replace("/", "_"):
                    found = j
                    break
        mapping.append(found)
    return mapping


# ---------------------------------------------------------------------------
# Trajectory recorder for current env outputs
# ---------------------------------------------------------------------------

class TrajectoryRecorder:
    """Buffers per-step data and saves to .npz on flush."""

    def __init__(self, save_dir: str):
        self._save_dir = save_dir
        os.makedirs(self._save_dir, exist_ok=True)
        self._traj_count = len([f for f in os.listdir(self._save_dir) if f.endswith(".npz")])
        self._buf: dict[str, list[np.ndarray]] = {}
        self._steps = 0

    def _append(self, key: str, arr: np.ndarray):
        self._buf.setdefault(key, []).append(np.asarray(arr).copy())

    def record(
        self,
        obs: dict,
        left_target_pos: np.ndarray,
        left_target_quat: np.ndarray,
        left_gripper: float,
        right_target_pos: np.ndarray,
        right_target_quat: np.ndarray,
        right_gripper: float,
        left_ee_pos: np.ndarray | None = None,
        left_ee_quat: np.ndarray | None = None,
        right_ee_pos: np.ndarray | None = None,
        right_ee_quat: np.ndarray | None = None,
        joint_commands: np.ndarray | None = None,
    ):
        left_action = np.concatenate([
            np.asarray(left_target_pos, dtype=np.float32).reshape(3),
            np.asarray(left_target_quat, dtype=np.float32).reshape(4),
            np.array([left_gripper], dtype=np.float32),
        ])
        right_action = np.concatenate([
            np.asarray(right_target_pos, dtype=np.float32).reshape(3),
            np.asarray(right_target_quat, dtype=np.float32).reshape(4),
            np.array([right_gripper], dtype=np.float32),
        ])
        self._append("left_actions", left_action)
        self._append("right_actions", right_action)
        self._append("actions", np.concatenate([left_action, right_action], axis=0))
        self._append("tactile", obs["tactile"])
        self._append("joint_pos", obs["joint_pos"])
        self._append("joint_vel", obs["joint_vel"])
        self._append("plug_pose", obs["plug_pose"])
        self._append("socket_pose", obs["socket_pose"])

        if "rgb" in obs:
            self._append("rgb", obs["rgb"])

        if left_ee_pos is not None and left_ee_quat is not None:
            self._append("left_eef_pos_quat", np.concatenate([
                np.asarray(left_ee_pos, dtype=np.float32).reshape(3),
                np.asarray(left_ee_quat, dtype=np.float32).reshape(4),
            ]))

        if right_ee_pos is not None and right_ee_quat is not None:
            self._append("right_eef_pos_quat", np.concatenate([
                np.asarray(right_ee_pos, dtype=np.float32).reshape(3),
                np.asarray(right_ee_quat, dtype=np.float32).reshape(4),
            ]))

        if joint_commands is not None:
            self._append("joint_commands", np.asarray(joint_commands, dtype=np.float32))

        self._steps += 1

    @property
    def num_steps(self) -> int:
        return self._steps

    def flush(self) -> str | None:
        if self._steps == 0:
            return None

        data = {k: np.stack(v, axis=0) for k, v in self._buf.items()}
        data["traj_lengths"] = np.array([self._steps], dtype=np.int32)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"traj_{self._traj_count:04d}_{ts}.npz"
        path = os.path.join(self._save_dir, fname)
        np.savez_compressed(path, **data)

        n = self._steps
        self._buf.clear()
        self._steps = 0
        self._traj_count += 1
        print(f"[SAVE] Trajectory saved: {path} ({n} steps, keys={sorted(data.keys())})", flush=True)
        return path


# ---------------------------------------------------------------------------
# Viser setup
# ---------------------------------------------------------------------------

def create_viser_server(port: int, cfg=None):
    import viser

    print(f"[viser] Creating ViserServer on port {port}...", flush=True)
    server = viser.ViserServer(port=port)
    print(f"[viser] *** Open http://localhost:{port} in your browser ***", flush=True)

    if cfg is not None and hasattr(server, "initial_camera"):
        try:
            eye = np.asarray(getattr(cfg, "camera_eye"), dtype=np.float64).reshape(3)
            target = np.asarray(getattr(cfg, "camera_target"), dtype=np.float64).reshape(3)
            server.initial_camera.position = eye
            server.initial_camera.look_at = target
            server.initial_camera.up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            print(
                f"[viser] Initial browser camera set: eye={eye.tolist()} target={target.tolist()}",
                flush=True,
            )
        except Exception as e:
            print(f"[viser] WARNING: Failed to set initial browser camera: {e}", flush=True)

    with server.gui.add_folder("Left Gripper"):
        left_btns = server.gui.add_button_group("Left gripper", options=["Open", "Close"])
        left_slider = server.gui.add_slider("Left fine", min=0.0, max=1.0, step=0.01, initial_value=1.0)

    with server.gui.add_folder("Right Gripper"):
        right_btns = server.gui.add_button_group("Right gripper", options=["Open", "Close"])
        right_slider = server.gui.add_slider("Right fine", min=0.0, max=1.0, step=0.01, initial_value=1.0)

    @left_btns.on_click
    def _(event):
        cur = shared.get_gripper("left")
        v = min(1.0, cur + 0.1) if event.target.value == "Open" else max(0.0, cur - 0.1)
        shared.set_gripper("left", v)
        left_slider.value = v

    @right_btns.on_click
    def _(event):
        cur = shared.get_gripper("right")
        v = min(1.0, cur + 0.1) if event.target.value == "Open" else max(0.0, cur - 0.1)
        shared.set_gripper("right", v)
        right_slider.value = v

    @left_slider.on_update
    def _(_):
        shared.set_gripper("left", left_slider.value)

    @right_slider.on_update
    def _(_):
        shared.set_gripper("right", right_slider.value)

    reset_btn = server.gui.add_button("Reset", color="red")

    @reset_btn.on_click
    def _(_):
        shared.request_reset()

    record_btn = server.gui.add_button("Start Recording", color="green")

    @record_btn.on_click
    def _(_):
        is_recording = shared.is_recording()
        shared.set_recording(not is_recording)
        record_btn.name = "Start Recording" if is_recording else "Stop Recording"
        record_btn.color = "green" if is_recording else "orange"

    dummy_img = np.zeros((96, 256, 3), dtype=np.uint8)
    with server.gui.add_folder("Tactile"):
        tac_handles = [
            server.gui.add_image(dummy_img, label=label, format="jpeg", jpeg_quality=80)
            for label in [
                "Left arm / left finger",
                "Left arm / right finger",
                "Right arm / left finger",
                "Right arm / right finger",
            ]
        ]

    with server.gui.add_folder("State", expand_by_default=False):
        state_md = server.gui.add_markdown("**Loading...**")

    with server.gui.add_folder("Sim Camera", expand_by_default=False):
        camera_img = server.gui.add_image(
            np.zeros((480, 640, 3), dtype=np.uint8),
            label="Camera Feed", format="jpeg", jpeg_quality=70,
        )

    server.scene.add_grid("/grid", width=2.0, height=2.0, position=(0.0, 0.0, -0.05))

    handles = {
        "server": server,
        "viser_urdf": None,
        "plug_mesh": None,
        "socket_mesh": None,
        "left_ee_gizmo": None,
        "right_ee_gizmo": None,
        "left_gripper_slider": left_slider,
        "right_gripper_slider": right_slider,
        "record_btn": record_btn,
        "tac_handles": tac_handles,
        "state_md": state_md,
        "camera_img": camera_img,
    }
    return server, handles


def load_scene_objects(server, handles, cfg):
    from viser.extras import ViserUrdf

    urdf_path = Path(os.path.expanduser(cfg.urdf_path))
    print(f"[viser] Loading URDF from {urdf_path}...", flush=True)
    try:
        viser_urdf = ViserUrdf(
            server,
            urdf_or_path=urdf_path,
            root_node_name="/robot",
            load_meshes=True,
            load_collision_meshes=False,
        )
        handles["viser_urdf"] = viser_urdf
        print("[viser] URDF loaded.", flush=True)
    except Exception as e:
        print(f"[viser] WARNING: URDF load failed: {e}", flush=True)

    asset_root = os.path.expanduser(getattr(cfg, "asset_root", ""))
    automate_dir = os.path.join(asset_root, "automate_scaled", "urdf")

    if cfg.enable_plug:
        import trimesh

        plug_urdf = os.path.join(automate_dir, f"{cfg.automate_asset_id}_plug.urdf")
        try:
            result = _extract_mesh_path_from_urdf(plug_urdf)
            if result:
                mesh_path, scale = result
                mesh = trimesh.load(mesh_path, force="mesh")
                mesh.apply_scale(scale)
                handles["plug_mesh"] = server.scene.add_mesh_trimesh(
                    "/plug",
                    mesh,
                    position=(0.0, 0.05, 0.0175),
                    wxyz=(0.5, -0.5, -0.5, -0.5),
                )
                print("[viser] Plug mesh loaded.", flush=True)
        except Exception as e:
            print(f"[viser] WARNING: Plug mesh load failed: {e}", flush=True)

    if cfg.enable_socket:
        import trimesh

        socket_urdf = os.path.join(automate_dir, f"{cfg.automate_asset_id}_socket.urdf")
        try:
            result = _extract_mesh_path_from_urdf(socket_urdf)
            if result:
                mesh_path, scale = result
                mesh = trimesh.load(mesh_path, force="mesh")
                mesh.apply_scale(scale)
                handles["socket_mesh"] = server.scene.add_mesh_trimesh(
                    "/socket",
                    mesh,
                    position=(0.0, -0.05, 0.123),
                    wxyz=(0.0, 0.0, 0.0, 1.0),
                )
                print("[viser] Socket mesh loaded.", flush=True)
        except Exception as e:
            print(f"[viser] WARNING: Socket mesh load failed: {e}", flush=True)


def create_ee_gizmo(server, handles, arm: str, ee_pos, ee_quat):
    label = f"/{arm}_ee_target"
    ee_gizmo = server.scene.add_transform_controls(
        label,
        scale=0.15,
        disable_axes=False,
        disable_sliders=False,
        disable_rotations=False,
        position=tuple(float(v) for v in ee_pos),
        wxyz=tuple(float(v) for v in ee_quat),
    )

    @ee_gizmo.on_update
    def _(event):
        pos = event.target.position
        quat = event.target.wxyz
        shared.set_target(arm, float(pos[0]), float(pos[1]), float(pos[2]))
        shared.set_target_quat(arm, float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))

    @ee_gizmo.on_drag_start
    def _(_):
        shared.set_dragging(arm, True)

    @ee_gizmo.on_drag_end
    def _(_):
        shared.set_dragging(arm, False)

    handles[f"{arm}_ee_gizmo"] = ee_gizmo
    shared.set_target(arm, *[float(v) for v in ee_pos])
    shared.set_target_quat(arm, *[float(v) for v in ee_quat])


def update_scene_fast(handles, obs, left_ee_pos, left_ee_quat, right_ee_pos, right_ee_quat, joint_name_map, urdf_joint_names):
    viser_urdf = handles["viser_urdf"]
    joint_pos = obs["joint_pos"]
    if viser_urdf is not None and joint_name_map is not None:
        cfg_array = np.zeros(len(urdf_joint_names), dtype=np.float32)
        for urdf_idx, dataset_idx in enumerate(joint_name_map):
            if dataset_idx is not None and dataset_idx < len(joint_pos):
                cfg_array[urdf_idx] = joint_pos[dataset_idx]
        viser_urdf.update_cfg(cfg_array)

    if handles["plug_mesh"] is not None:
        plug_pose = obs["plug_pose"]
        handles["plug_mesh"].position = tuple(float(v) for v in plug_pose[:3])
        handles["plug_mesh"].wxyz = tuple(float(v) for v in plug_pose[3:7])

    if handles["socket_mesh"] is not None:
        socket_pose = obs["socket_pose"]
        if np.any(np.abs(socket_pose) > 1e-8):
            handles["socket_mesh"].position = tuple(float(v) for v in socket_pose[:3])
            handles["socket_mesh"].wxyz = tuple(float(v) for v in socket_pose[3:7])

    if handles["left_ee_gizmo"] is not None and not shared.is_dragging("left"):
        handles["left_ee_gizmo"].position = tuple(float(v) for v in left_ee_pos)
        handles["left_ee_gizmo"].wxyz = tuple(float(v) for v in left_ee_quat)

    if handles["right_ee_gizmo"] is not None and not shared.is_dragging("right"):
        handles["right_ee_gizmo"].position = tuple(float(v) for v in right_ee_pos)
        handles["right_ee_gizmo"].wxyz = tuple(float(v) for v in right_ee_quat)


def update_scene_slow(handles, obs, left_ee, right_ee, left_gripper, right_gripper, step, recording, rec_steps):
    tactile = obs["tactile"]
    tac_maxes = []
    for i, handle in enumerate(handles["tac_handles"]):
        grid = tactile[i]
        tac_maxes.append(float(grid.max()))
        try:
            handle.image = tactile_to_rgb(grid)
        except Exception:
            pass

    if "rgb" in obs:
        try:
            handles["camera_img"].image = obs["rgb"]
        except Exception:
            pass

    rec_status = f"  **REC** ({rec_steps} steps)" if recording else ""
    tac_str = ", ".join(f"{m:.4f}" for m in tac_maxes)
    handles["state_md"].content = (
        f"**Left EE:** ({left_ee[0]:.4f}, {left_ee[1]:.4f}, {left_ee[2]:.4f})  \n"
        f"**Right EE:** ({right_ee[0]:.4f}, {right_ee[1]:.4f}, {right_ee[2]:.4f})  \n"
        f"**Left gripper:** {left_gripper:.2f}  \n"
        f"**Right gripper:** {right_gripper:.2f}  \n"
        f"**Tactile max [4 pads]:** [{tac_str}]  \n"
        f"**Step:** {step}{rec_status}"
    )


# ---------------------------------------------------------------------------
# Pose-mode IK
# ---------------------------------------------------------------------------

def upgrade_ik_to_pose_mode(ik_ctrl):
    from isaaclab.controllers.differential_ik import DifferentialIKController
    from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg

    pose_cfg = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=False,
        ik_method="dls",
        ik_params={"lambda_val": 0.05},
    )
    ik_ctrl._ik_controller = DifferentialIKController(cfg=pose_cfg, num_envs=1, device=ik_ctrl._device)


def compute_pose(ik_ctrl, target_pos, target_quat, gripper):
    from aloha_ik_controller import GRIPPER_CLOSED, GRIPPER_OPEN

    target_pos = np.asarray(target_pos, dtype=np.float32).reshape(3)
    target_quat = np.asarray(target_quat, dtype=np.float32).reshape(4)

    command = np.concatenate([target_pos, target_quat])
    command_t = torch.tensor(command, dtype=torch.float32, device=ik_ctrl._device).unsqueeze(0)

    ee_pos_w = ik_ctrl._robot.data.body_pos_w[:, ik_ctrl._ee_body_idx]
    ee_quat_w = ik_ctrl._robot.data.body_quat_w[:, ik_ctrl._ee_body_idx]
    current_arm_pos = ik_ctrl._robot.data.joint_pos[:, ik_ctrl._arm_joint_ids]
    jacobian = ik_ctrl._robot.root_physx_view.get_jacobians()[:, ik_ctrl._jacobi_body_idx, :, ik_ctrl._jacobi_joint_ids]

    ik_ctrl._ik_controller.set_command(command=command_t)
    joint_pos_des = ik_ctrl._ik_controller.compute(ee_pos_w, ee_quat_w, jacobian, current_arm_pos)

    all_joint_pos = ik_ctrl._robot.data.joint_pos[0, ik_ctrl._dataset_joint_ids].clone()
    action = all_joint_pos.detach().cpu().numpy().astype(np.float32)

    for i, arm_jid in enumerate(ik_ctrl._arm_joint_ids):
        slot = ik_ctrl._artjid_to_dataset_slot[arm_jid]
        action[slot] = float(joint_pos_des[0, i].detach().cpu())

    gripper = float(np.clip(gripper, 0.0, 1.0))
    gripper_pos = GRIPPER_CLOSED + gripper * (GRIPPER_OPEN - GRIPPER_CLOSED)
    left_slot, right_slot = ik_ctrl._gripper_slots
    action[left_slot] = gripper_pos
    action[right_slot] = -gripper_pos

    return action


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    port = int(args.port)

    cfg = AlohaTactileEnvCfg(
        headless=bool(getattr(args, "headless", True)),
        device=str(getattr(args, "device", "cuda:0")),
        enable_camera=True,
    )

    if not cfg.headless:
        try:
            from isaacsim.core.utils.viewports import set_camera_view
            set_camera_view(list(cfg.camera_eye), list(cfg.camera_target))
        except Exception as e:
            print(f"[WARN] Failed to set viewport camera view: {e}", flush=True)

    server, handles = create_viser_server(port, cfg)

    cfg = AlohaTactileEnvCfg(
        headless=bool(getattr(args, "headless", True)),
        device=str(getattr(args, "device", "cuda:0")),
        enable_camera=True,
    )

    handles["state_md"].content = "**Loading Isaac Sim environment...**"
    patch_env_spawn_objects_for_socket_articulation_fix()
    env = AlohaTactileEnv(cfg, simulation_app=simulation_app)
    obs, _ = env.reset()

    left_ik = AlohaArmIKController(robot=env._robot, device=cfg.device, arm="left")
    right_ik = AlohaArmIKController(robot=env._robot, device=cfg.device, arm="right")
    upgrade_ik_to_pose_mode(left_ik)
    upgrade_ik_to_pose_mode(right_ik)

    aloha_init = np.array([
        0.0, -0.16, 1.15, 0.0, -0.5, 0.0, 0.057, -0.057,
        0.0, -0.16, 1.15, 0.0, -0.5, 0.0, 0.057, -0.057,
    ], dtype=np.float32)
    left_arm_init = aloha_init[:8].copy()
    right_arm_init = aloha_init[8:].copy()
    init_joint_pos = aloha_init.copy()
    left_gripper_init = 1.0
    right_gripper_init = 1.0

    warmup_steps = 50
    handles["state_md"].content = "**Running warmup steps...**"
    for _ in range(warmup_steps):
        obs, _, _, _, _ = env.step(init_joint_pos)

    left_ee_target_pos = np.array([0.0, -0.167, 0.074], dtype=np.float64)
    right_ee_target_pos = np.array([0.0, 0.167, 0.074], dtype=np.float64)
    left_ee_quat_init = left_ik.get_ee_quat()
    right_ee_quat_init = right_ik.get_ee_quat()

    save_dir = args.save_dir or os.path.join(os.path.dirname(__file__), "output", "trajectories")
    recorder = TrajectoryRecorder(save_dir)
    print(f"[INFO] Trajectories will be saved to: {save_dir}", flush=True)

    handles["state_md"].content = "**Loading 3D models...**"
    load_scene_objects(server, handles, cfg)
    create_ee_gizmo(server, handles, "left", left_ee_target_pos, left_ee_quat_init)
    create_ee_gizmo(server, handles, "right", right_ee_target_pos, right_ee_quat_init)
    shared.set_target("left", *[float(v) for v in left_ee_target_pos])
    shared.set_target_quat("left", *[float(v) for v in left_ee_quat_init])
    shared.set_target("right", *[float(v) for v in right_ee_target_pos])
    shared.set_target_quat("right", *[float(v) for v in right_ee_quat_init])
    shared.set_gripper("left", left_gripper_init)
    shared.set_gripper("right", right_gripper_init)
    handles["left_gripper_slider"].value = left_gripper_init
    handles["right_gripper_slider"].value = right_gripper_init

    urdf_joint_names = ()
    joint_name_map = None
    if handles["viser_urdf"] is not None:
        urdf_joint_names = handles["viser_urdf"].get_actuated_joint_names()
        joint_name_map = build_joint_name_map(urdf_joint_names)

    handles["state_md"].content = "**Ready. Drag either gizmo to control the bimanual arms.**"

    print('\n' + '=' * 72, flush=True)
    print(f'[VISER URL] http://localhost:{port}', flush=True)
    print(f'[VISER URL] Open this in your browser: http://localhost:{port}', flush=True)
    print('=' * 72 + '\n', flush=True)

    step = 0
    slow_interval = max(1, int(args.slow_interval))
    was_recording = False

    while simulation_app.is_running():
        recording = shared.is_recording()

        if was_recording and not recording:
            path = recorder.flush()
            if path:
                handles["state_md"].content = f"**Saved!** {os.path.basename(path)}"
        was_recording = recording

        if shared.consume_reset():
            if recorder.num_steps > 0:
                recorder.flush()
            shared.set_recording(False)
            handles["record_btn"].name = "Start Recording"
            handles["record_btn"].color = "green"
            was_recording = False

            obs, _ = env.reset()
            for _ in range(warmup_steps):
                obs, _, _, _, _ = env.step(init_joint_pos)

            left_pos = left_ee_target_pos.copy()
            right_pos = right_ee_target_pos.copy()
            left_quat = left_ik.get_ee_quat().astype(np.float64)
            right_quat = right_ik.get_ee_quat().astype(np.float64)

            left_delta = np.random.uniform(-np.radians(15), np.radians(15))
            right_delta = np.random.uniform(-np.radians(15), np.radians(15))
            left_dq = np.array([np.cos(left_delta / 2.0), np.sin(left_delta / 2.0), 0.0, 0.0], dtype=np.float64)
            right_dq = np.array([np.cos(right_delta / 2.0), np.sin(right_delta / 2.0), 0.0, 0.0], dtype=np.float64)
            left_quat = _quat_mul_wxyz(left_dq, left_quat)
            right_quat = _quat_mul_wxyz(right_dq, right_quat)
            left_quat /= np.linalg.norm(left_quat)
            right_quat /= np.linalg.norm(right_quat)

            shared.set_target("left", *[float(v) for v in left_pos])
            shared.set_target_quat("left", *[float(v) for v in left_quat])
            shared.set_target("right", *[float(v) for v in right_pos])
            shared.set_target_quat("right", *[float(v) for v in right_quat])
            shared.set_gripper("left", left_gripper_init)
            shared.set_gripper("right", right_gripper_init)
            handles["left_gripper_slider"].value = left_gripper_init
            handles["right_gripper_slider"].value = right_gripper_init
            if handles["left_ee_gizmo"] is not None:
                handles["left_ee_gizmo"].position = tuple(float(v) for v in left_pos)
                handles["left_ee_gizmo"].wxyz = tuple(float(v) for v in left_quat)
            if handles["right_ee_gizmo"] is not None:
                handles["right_ee_gizmo"].position = tuple(float(v) for v in right_pos)
                handles["right_ee_gizmo"].wxyz = tuple(float(v) for v in right_quat)
            step = 0

        left_target_pos = np.array(shared.get_target("left"), dtype=np.float32)
        left_target_quat = np.array(shared.get_target_quat("left"), dtype=np.float32)
        right_target_pos = np.array(shared.get_target("right"), dtype=np.float32)
        right_target_quat = np.array(shared.get_target_quat("right"), dtype=np.float32)
        left_gripper = shared.get_gripper("left")
        right_gripper = shared.get_gripper("right")

        left_action = compute_pose(left_ik, left_target_pos, left_target_quat, left_gripper)
        right_action = compute_pose(right_ik, right_target_pos, right_target_quat, right_gripper)
        action = np.concatenate([left_action[:8], right_action[8:]], axis=0).astype(np.float32)
        obs, _, _, _, _ = env.step(action)

        left_ee = left_ik.get_ee_pos()
        left_ee_quat = left_ik.get_ee_quat()
        right_ee = right_ik.get_ee_pos()
        right_ee_quat = right_ik.get_ee_quat()

        if recording:
            recorder.record(
                obs,
                left_target_pos,
                left_target_quat,
                left_gripper,
                right_target_pos,
                right_target_quat,
                right_gripper,
                left_ee_pos=left_ee,
                left_ee_quat=left_ee_quat,
                right_ee_pos=right_ee,
                right_ee_quat=right_ee_quat,
                joint_commands=action,
            )

        update_scene_fast(
            handles,
            obs,
            left_ee,
            left_ee_quat,
            right_ee,
            right_ee_quat,
            joint_name_map,
            urdf_joint_names,
        )

        if step % slow_interval == 0:
            update_scene_slow(
                handles,
                obs,
                left_ee,
                right_ee,
                left_gripper,
                right_gripper,
                step,
                recording,
                recorder.num_steps,
            )

        if step % 120 == 0:
            tac = [float(obs["tactile"][i].max()) for i in range(4)]
            rec_tag = f" REC({recorder.num_steps})" if recording else ""
            print(
                f"[step {step:06d}]"
                f" L_ee=({left_ee[0]:.3f},{left_ee[1]:.3f},{left_ee[2]:.3f})"
                f" R_ee=({right_ee[0]:.3f},{right_ee[1]:.3f},{right_ee[2]:.3f})"
                f" L_grip={left_gripper:.2f}"
                f" R_grip={right_gripper:.2f}"
                f" tactile={','.join(f'{v:.4f}' for v in tac)}"
                f"{rec_tag}",
                flush=True,
            )

        step += 1

    if recorder.num_steps > 0:
        recorder.flush()

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
