# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class WarpSdfTactileSensorData:
    """Data container for :class:`WarpSdfTactileSensor`."""

    tactile_points_w: torch.Tensor | None = None
    """Flattened tactile points per env: shape (E, S*P, 4) with columns [x, y, z, fn]."""

    tactile_points_w_per_sensor: torch.Tensor | None = None
    """Tactile points per env and sensor: shape (E, S, P, 4)."""
