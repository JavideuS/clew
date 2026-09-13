"""Shared 2D geometry primitives, reused across ordering.py, recovery.py,
dispatch.py, and coordinator_node.py

Math Only (no ROS dependencies)
"""

from __future__ import annotations

import math

Point = tuple[float, float]
Pose = tuple[float, float, float]


def distance(a: Point, b: Point) -> float:
    """Euclidean distance between two (x, y) points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def heading_to(a: Point, b: Point) -> float:
    """Heading (radians) of the straight line from `a` to `b`."""
    return math.atan2(b[1] - a[1], b[0] - a[0])


def angle_diff(a: float, b: float) -> float:
    """Signed difference a - b, wrapped to (-pi, pi] -- the standard
    atan2(sin, cos) trick, so callers never have to handle the wraparound
    at +-pi themselves.
    """
    d = a - b
    return math.atan2(math.sin(d), math.cos(d))


def yaw_to_quat_zw(theta: float) -> tuple[float, float]:
    """(z, w) quaternion components for a planar heading -- x and y are
    always 0 for a ground robot's yaw-only orientation (roll = pitch = 0).
    """
    return math.sin(theta / 2.0), math.cos(theta / 2.0)


def quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    """Yaw extracted from a full quaternion. Roll/pitch are ignored --
    correct for a ground robot, where a real-world quaternion may carry
    small roll/pitch noise but only yaw is ever meaningful.
    """
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
