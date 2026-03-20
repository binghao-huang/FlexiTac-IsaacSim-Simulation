"""Differential IK controller for the ALOHA arms.

Accepts (x, y, z) position + gripper scalar, produces 16-dim joint targets
compatible with AlohaTactileEnv.step().

Requires IsaacLab imports (must be loaded after SimulationApp is created).
"""

from __future__ import annotations

import numpy as np
import torch

# These imports require SimulationApp to exist already
from isaaclab.assets import Articulation
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg

# Dataset joint order (same as aloha_tactile_env.py)
DATASET_JOINT_ORDER = [
    "left/waist",
    "left/shoulder",
    "left/elbow",
    "left/forearm_roll",
    "left/wrist_angle",
    "left/wrist_rotate",
    "left/left_finger",
    "left/right_finger",
    "right/waist",
    "right/shoulder",
    "right/elbow",
    "right/forearm_roll",
    "right/wrist_angle",
    "right/wrist_rotate",
    "right/left_finger",
    "right/right_finger",
]

# Left arm: 6 revolute joints for IK
LEFT_ARM_JOINT_NAMES = [
    "left/waist",
    "left/shoulder",
    "left/elbow",
    "left/forearm_roll",
    "left/wrist_angle",
    "left/wrist_rotate",
]

# Left arm: 2 gripper joints
LEFT_GRIPPER_JOINT_NAMES = [
    "left/left_finger",
    "left/right_finger",
]

# Right arm: 6 revolute joints for IK
RIGHT_ARM_JOINT_NAMES = [
    "right/waist",
    "right/shoulder",
    "right/elbow",
    "right/forearm_roll",
    "right/wrist_angle",
    "right/wrist_rotate",
]

# Right arm: 2 gripper joints
RIGHT_GRIPPER_JOINT_NAMES = [
    "right/left_finger",
    "right/right_finger",
]

# Gripper prismatic limits (meters)
GRIPPER_OPEN = 0.042
GRIPPER_CLOSED = 0.0

# Per-arm constants lookup
_ARM_CONFIGS = {
    "left": {
        "arm_joints": LEFT_ARM_JOINT_NAMES,
        "gripper_joints": LEFT_GRIPPER_JOINT_NAMES,
        "gripper_slots": (6, 7),  # dataset order indices
    },
    "right": {
        "arm_joints": RIGHT_ARM_JOINT_NAMES,
        "gripper_joints": RIGHT_GRIPPER_JOINT_NAMES,
        "gripper_slots": (14, 15),  # dataset order indices
    },
}


def _fuzzy_find_joint(robot: Articulation, name: str) -> int:
    """Fuzzy-match a dataset joint name to an articulation DOF index."""
    joint_names = [str(n) for n in robot.joint_names]
    joint_names_l = [n.lower() for n in joint_names]
    t = name.replace("/", "_").lower()
    if t in joint_names_l:
        return joint_names_l.index(t)
    cands = [i for i, n in enumerate(joint_names_l) if n.endswith(t) or (t in n)]
    if not cands:
        raise RuntimeError(f"Cannot map joint '{name}' to articulation DOFs: {joint_names}")
    return sorted(cands, key=lambda i: len(joint_names_l[i]))[0]


def _fuzzy_find_body(robot: Articulation, name: str) -> int:
    """Fuzzy-match a body name to an articulation body index."""
    body_names = [str(n) for n in robot.body_names]
    body_names_l = [n.lower() for n in body_names]
    t = name.replace("/", "_").lower()
    if t in body_names_l:
        return body_names_l.index(t)
    cands = [i for i, n in enumerate(body_names_l) if t in n]
    if not cands:
        raise RuntimeError(f"Cannot find body '{name}' in: {body_names}")
    return sorted(cands, key=lambda i: len(body_names_l[i]))[0]


class AlohaArmIKController:
    """IK controller for one ALOHA arm (left or right).

    Accepts world-frame (x, y, z) + gripper scalar, produces 16-dim joint
    position targets in dataset order. The other arm's joints are held at
    their current positions.
    """

    def __init__(self, robot: Articulation, device: str = "cuda:0", arm: str = "left"):
        if arm not in _ARM_CONFIGS:
            raise ValueError(f"arm must be 'left' or 'right', got '{arm}'")

        self._robot = robot
        self._device = device
        self._arm = arm
        cfg_arm = _ARM_CONFIGS[arm]

        # 1. Resolve arm revolute joint indices (6 DOFs for IK)
        self._arm_joint_ids = [_fuzzy_find_joint(robot, n) for n in cfg_arm["arm_joints"]]
        self._arm_joint_names = [robot.joint_names[i] for i in self._arm_joint_ids]
        print(f"[IK-{arm}] Arm joints: {list(zip(self._arm_joint_names, self._arm_joint_ids))}", flush=True)

        # 2. Resolve gripper joint indices
        self._gripper_joint_ids = [_fuzzy_find_joint(robot, n) for n in cfg_arm["gripper_joints"]]
        self._gripper_slots = cfg_arm["gripper_slots"]
        print(f"[IK-{arm}] Gripper joints: {self._gripper_joint_ids} (slots {self._gripper_slots})", flush=True)

        # 3. Resolve EE body index
        self._ee_body_idx = _fuzzy_find_body(robot, "ee_gripper_link")
        ee_name = robot.body_names[self._ee_body_idx]
        # Make sure it's the correct arm's EE
        if arm not in ee_name.lower():
            body_names_l = [n.lower() for n in robot.body_names]
            arm_ee = [i for i, n in enumerate(body_names_l) if arm in n and "ee_gripper" in n]
            if arm_ee:
                self._ee_body_idx = arm_ee[0]
                ee_name = robot.body_names[self._ee_body_idx]
        print(f"[IK-{arm}] EE body: '{ee_name}' (idx={self._ee_body_idx})", flush=True)

        # 4. Jacobian index correction (fixed-base vs floating-base)
        if robot.is_fixed_base:
            self._jacobi_body_idx = self._ee_body_idx - 1
            self._jacobi_joint_ids = self._arm_joint_ids
        else:
            self._jacobi_body_idx = self._ee_body_idx
            self._jacobi_joint_ids = [i + 6 for i in self._arm_joint_ids]
        print(
            f"[IK-{arm}] is_fixed_base={robot.is_fixed_base}"
            f" jacobi_body_idx={self._jacobi_body_idx}"
            f" jacobi_joint_ids={self._jacobi_joint_ids}",
            flush=True,
        )

        # 5. Create DifferentialIKController
        ik_cfg = DifferentialIKControllerCfg(
            command_type="position",
            use_relative_mode=False,
            ik_method="dls",
            ik_params={"lambda_val": 0.05},
        )
        self._ik_controller = DifferentialIKController(cfg=ik_cfg, num_envs=1, device=device)

        # 6. Build full dataset-order joint ID mapping
        self._dataset_joint_ids = [_fuzzy_find_joint(robot, n) for n in DATASET_JOINT_ORDER]

        # 7. Build reverse map: articulation joint id -> dataset slot index
        self._artjid_to_dataset_slot: dict[int, int] = {}
        for slot, art_id in enumerate(self._dataset_joint_ids):
            self._artjid_to_dataset_slot[art_id] = slot

    @property
    def arm(self) -> str:
        return self._arm

    @property
    def ee_body_idx(self) -> int:
        return self._ee_body_idx

    def get_ee_pos(self) -> np.ndarray:
        """Return current (x, y, z) of the arm EE in world frame."""
        return (
            self._robot.data.body_pos_w[0, self._ee_body_idx]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )

    def get_ee_quat(self) -> np.ndarray:
        """Return current quaternion (w, x, y, z) of the arm EE in world frame."""
        return (
            self._robot.data.body_quat_w[0, self._ee_body_idx]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )

    def compute(self, target_pos: np.ndarray, gripper: float) -> np.ndarray:
        """Compute 16-dim joint position targets from (x, y, z) + gripper.

        Args:
            target_pos: (3,) absolute world-frame position target for EE.
            gripper: float in [0, 1], 0=closed, 1=open.

        Returns:
            (16,) joint position targets in dataset order.
        """
        target_pos = np.asarray(target_pos, dtype=np.float32).reshape(3)
        target_t = torch.tensor(target_pos, dtype=torch.float32, device=self._device).unsqueeze(0)

        # Current EE pose in world frame
        ee_pos_w = self._robot.data.body_pos_w[:, self._ee_body_idx]  # (1, 3)
        ee_quat_w = self._robot.data.body_quat_w[:, self._ee_body_idx]  # (1, 4)

        # Current arm joint positions (6 revolute DOFs)
        current_arm_pos = self._robot.data.joint_pos[:, self._arm_joint_ids]  # (1, 6)

        # Jacobian for the arm (world frame)
        jacobian = self._robot.root_physx_view.get_jacobians()[
            :, self._jacobi_body_idx, :, self._jacobi_joint_ids
        ]  # (1, 6, 6)

        # Set IK command (absolute world position)
        self._ik_controller.set_command(command=target_t, ee_quat=ee_quat_w)

        # Compute desired arm joint positions
        joint_pos_des = self._ik_controller.compute(
            ee_pos_w, ee_quat_w, jacobian, current_arm_pos
        )  # (1, 6)

        # Build full 16-dim action: start from current joint positions
        all_joint_pos = self._robot.data.joint_pos[0, self._dataset_joint_ids].clone()
        action = all_joint_pos.detach().cpu().numpy().astype(np.float32)

        # Fill in IK result for arm revolute joints
        for i, arm_jid in enumerate(self._arm_joint_ids):
            slot = self._artjid_to_dataset_slot[arm_jid]
            action[slot] = float(joint_pos_des[0, i].detach().cpu())

        # Fill in gripper
        gripper = float(np.clip(gripper, 0.0, 1.0))
        gripper_pos = GRIPPER_CLOSED + gripper * (GRIPPER_OPEN - GRIPPER_CLOSED)
        left_slot, right_slot = self._gripper_slots
        action[left_slot] = gripper_pos        # left_finger (positive)
        action[right_slot] = -gripper_pos      # right_finger (negative)

        return action


# Backwards-compatible alias
AlohaLeftArmIKController = AlohaArmIKController
