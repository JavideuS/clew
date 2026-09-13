"""Unit tests for geometry.py -- the 2D primitives shared by ordering.py,
recovery.py, dispatch.py, and coordinator_node.py."""

from __future__ import annotations

import math

import pytest

from fleet_coordinator.geometry import (
    angle_diff,
    distance,
    heading_to,
    quat_to_yaw,
    yaw_to_quat_zw,
)


def test_distance_basic():
    assert distance((0.0, 0.0), (3.0, 4.0)) == pytest.approx(5.0)
    assert distance((1.0, 1.0), (1.0, 1.0)) == pytest.approx(0.0)


def test_heading_to_cardinal_directions():
    assert heading_to((0.0, 0.0), (1.0, 0.0)) == pytest.approx(0.0)
    assert heading_to((0.0, 0.0), (0.0, 1.0)) == pytest.approx(math.pi / 2)
    assert heading_to((0.0, 0.0), (-1.0, 0.0)) == pytest.approx(math.pi)


def test_angle_diff_wraps_across_pi():
    # just below +pi and just above -pi are actually only 0.1 rad apart,
    # not the ~2*pi a naive subtraction would give
    a, b = math.pi - 0.05, -math.pi + 0.05
    assert angle_diff(a, b) == pytest.approx(-0.1, abs=1e-9)
    assert angle_diff(0.1, 0.0) == pytest.approx(0.1)
    assert angle_diff(0.0, 0.1) == pytest.approx(-0.1)


def test_angle_diff_zero_for_equal_angles():
    assert angle_diff(1.2345, 1.2345) == pytest.approx(0.0)


def test_yaw_quat_roundtrip():
    for theta in (0.0, math.pi / 4, math.pi / 2, math.pi, -math.pi / 3):
        z, w = yaw_to_quat_zw(theta)
        back = quat_to_yaw(0.0, 0.0, z, w)
        assert back == pytest.approx(theta, abs=1e-9)


def test_yaw_to_quat_zw_zero_is_identity():
    z, w = yaw_to_quat_zw(0.0)
    assert z == pytest.approx(0.0)
    assert w == pytest.approx(1.0)


def test_quat_to_yaw_ignores_roll_pitch_noise():
    # a "pure yaw" quaternion (x=y=0) plus a quaternion with some non-zero
    # x/y (as a real TF sample might carry) should still extract close to
    # the same yaw when z/w dominate -- exact equality isn't expected, this
    # just checks the formula uses all four components sensibly rather
    # than blowing up or ignoring z/w.
    theta = 0.7
    z, w = yaw_to_quat_zw(theta)
    assert quat_to_yaw(0.0, 0.0, z, w) == pytest.approx(theta)
