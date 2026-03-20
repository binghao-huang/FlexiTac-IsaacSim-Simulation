import numpy as np

def _xyzw_to_wxyz(q):
    return (float(q[3]), float(q[0]), float(q[1]), float(q[2]))

def _look_at_quat(eye, target):
    """Compute wxyz quaternion for a camera at *eye* looking at *target* (Z-up world)."""
    eye_a, tgt_a = np.asarray(eye, np.float64), np.asarray(target, np.float64)

    fwd = tgt_a - eye_a
    fwd /= np.linalg.norm(fwd) + 1e-12
    right = np.cross(fwd, [0, 0, 1])
    right /= np.linalg.norm(right) + 1e-12
    up = np.cross(right, fwd)
    up /= np.linalg.norm(up) + 1e-12

    R = np.column_stack([fwd, np.cross(up, fwd), up])

    # Shepperd's method: rotation matrix → wxyz quaternion
    tr = np.trace(R)
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s

    n = np.sqrt(w * w + x * x + y * y + z * z)
    return (float(w / n), float(x / n), float(y / n), float(z / n))



def _resolve_joint_ids(articulation, names: list[str]) -> list[int]:
    """Map dataset joint names to articulation DOF indices (fuzzy)."""
    joint_names_l = [str(n).lower() for n in articulation.joint_names]
    ids = []
    for token in names:
        t = token.replace("/", "_").lower()
        if t in joint_names_l:
            ids.append(joint_names_l.index(t))
            continue
        cands = [i for i, n in enumerate(joint_names_l) if n.endswith(t) or t in n]
        if not cands:
            raise RuntimeError(f"Failed to map dataset joint '{token}'")
        ids.append(min(cands, key=lambda i: len(joint_names_l[i])))
    return ids


def _to_numpy_1d(x, expected: int):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x).reshape(-1)
    if x.size != expected:
        raise ValueError(f"Unexpected size {x.size} (expected {expected})")
    return x

def _to_numpy_1d(x, expected: int):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x).reshape(-1)
    if x.size != expected:
        raise ValueError(f"Unexpected size {x.size} (expected {expected})")
    return x

