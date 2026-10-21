"""Explicit bridge between a latched source/planning frame and the live mount.

AutoData's source-segment transform uses the object pose at segment start. Raw
floating-mount records must first be expressed in one fixed source frame. A
future generator must latch its target frame at segment/planning start and use
``to_live_mount`` before sending each waypoint to the live R5a IK controller.
This module moves coordinates only, never the robot or its physical root.
"""
import numpy as np

def to_fixed_frame(world_fixed,world_mount,poses_in_live_mount):
    return np.linalg.inv(world_fixed)@world_mount@poses_in_live_mount

def to_live_mount(world_fixed,world_mount,poses_in_fixed_frame):
    return np.linalg.inv(world_mount)@world_fixed@poses_in_fixed_frame
