# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0
"""Signed viewpoint clearance remains correct for rotated door panels."""
import numpy as np
import pytest

from arx_r5_isaac_sim_bringup.door_view_guard import point_box_clearance


def test_inside_panel_is_negative():
    assert point_box_clearance([0, 0, 0], np.eye(4), [.4, .02, .4]) == pytest.approx(-.02)


def test_rotated_panel_clearance():
    transform = np.eye(4)
    transform[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    transform[:3, 3] = [1, 2, 3]
    assert point_box_clearance([.95, 2, 3], transform, [.4, .02, .4]) == pytest.approx(.03)


def test_nonfinite_viewpoint_rejected():
    with pytest.raises(ValueError):
        point_box_clearance([np.nan, 0, 0], np.eye(4), [.4, .02, .4])
