import os

import xml.etree.ElementTree as ET

def _parse_elastomer_origins(urdf_path: str) -> dict:
    """Return ``{"left": (xyz, rpy), "right": (xyz, rpy)}`` from URDF elastomer joints."""
    urdf_path = os.path.expanduser(str(urdf_path))
    if not os.path.isfile(urdf_path):
        raise FileNotFoundError(urdf_path)

    out = {}
    for joint in ET.parse(urdf_path).getroot().findall("joint"):
        name = joint.get("name", "").lower()
        if "elastomer_joint_left" in name:
            key = "left"
        elif "elastomer_joint_right" in name:
            key = "right"
        else:
            continue

        origin = joint.find("origin")
        if origin is None:
            continue
        try:
            xyz = tuple(float(v) for v in origin.get("xyz", "0 0 0").split())
            rpy = tuple(float(v) for v in origin.get("rpy", "0 0 0").split())
            if len(xyz) != 3 or len(rpy) != 3:
                continue
        except (ValueError, TypeError):
            continue

        out[key] = (xyz, rpy)
        if len(out) == 2:
            break

    missing = {"left", "right"} - out.keys()
    if missing:
        raise RuntimeError(f"Missing elastomer joint origins in URDF: {sorted(missing)}")
    return out


def _infer_arm(link_path: str) -> str | None:
    lp = link_path.lower()
    if "left_arm_" in lp or "/left/" in lp:
        return "left"
    if "right_arm_" in lp or "/right/" in lp:
        return "right"
    return None


def _infer_finger(link_path: str) -> str | None:
    lp = link_path.lower()
    if "elastomer_left" in lp or "_left_finger_link" in lp:
        return "left_finger"
    if "elastomer_right" in lp or "_right_finger_link" in lp:
        return "right_finger"
    return None


def _sensor_slot(link_path: str) -> int | None:
    """Canonical slot: 0=L/L, 1=L/R, 2=R/L, 3=R/R."""
    arm, finger = _infer_arm(link_path), _infer_finger(link_path)
    if arm is None or finger is None:
        return None
    return (0 if arm == "left" else 2) + (0 if finger == "left_finger" else 1)
