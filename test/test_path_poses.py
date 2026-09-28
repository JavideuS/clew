"""Unit tests for path_poses.py -- dispatch.py's pure point/heading
helpers. RobotDispatcher itself needs rclpy/nav2_msgs (not importable in a
plain Python environment) and can't be exercised here -- see dispatch.py's
module docstring for what's still unverified against a real ROS2 instance.
"""

from __future__ import annotations

import math

from clew.path_poses import (
    dedupe_consecutive,
    derive_headings,
    nearest_index,
)


def test_dedupe_consecutive_collapses_runs():
    assert dedupe_consecutive([(0, 0), (1, 0), (1, 0), (1, 0), (2, 0)]) == [
        (0, 0),
        (1, 0),
        (2, 0),
    ]


def test_dedupe_consecutive_keeps_non_adjacent_repeats():
    # (0, 0) appears twice but not back-to-back -- not a "wait", a real revisit.
    assert dedupe_consecutive([(0, 0), (1, 0), (0, 0)]) == [(0, 0), (1, 0), (0, 0)]


def test_derive_headings_empty():
    assert derive_headings([], 0.0, 0.0) == []


def test_derive_headings_single_point_collapse_uses_start_theta():
    # whole released prefix collapsed to one point -- "barely started", not "arrived"
    result = derive_headings([(1.0, 1.0), (1.0, 1.0)], start_theta=0.5, goal_theta=2.5)
    assert result == [(1.0, 1.0, 0.5)]


def test_derive_headings_pins_first_and_last_to_mission_theta():
    # goal_theta given (the release is genuinely the full/final path) --
    # both endpoints pinned.
    result = derive_headings([(0, 0), (1, 0), (2, 0)], start_theta=0.1, goal_theta=0.9)
    assert result[0] == (0, 0, 0.1)
    assert result[-1] == (2, 0, 0.9)


def test_derive_headings_partial_release_does_not_pin_last_point():
    # goal_theta=None (a partial/intermediate release -- gating hasn't
    # released the rest yet): pinning a non-final endpoint to the far-off
    # mission goal heading makes the controller treat every gated pause as
    # a real arrival and spin trying to align to it. The last point should
    # get the same backward-tangent heading as any other interior point
    # instead.
    result = derive_headings([(0, 0), (1, 0), (1, 1)], start_theta=0.1, goal_theta=None)
    assert result[0] == (0, 0, 0.1)  # start still pinned
    # last point (1,1): arrived via (1,0) -> (1,1), i.e. +y -- NOT pinned
    # to any mission goal_theta.
    assert math.isclose(result[-1][2], math.pi / 2, abs_tol=1e-9)


def test_derive_headings_middle_points_use_tangent():
    # Heading convention: a middle point's heading is the *arrival*
    # direction, the tangent from the previous point -- not a look-ahead to
    # the next one.
    result = derive_headings(
        [(0, 0), (1, 0), (1, 1), (1, 2)], start_theta=0.0, goal_theta=0.0
    )
    # index 1: arrived via (0,0) -> (1,0), i.e. +x
    assert math.isclose(result[1][2], 0.0, abs_tol=1e-9)
    # index 2: arrived via (1,0) -> (1,1), i.e. +y
    assert math.isclose(result[2][2], math.pi / 2, abs_tol=1e-9)


def test_derive_headings_dedupes_before_computing_tangent():
    # a "wait" step (duplicate point) must not produce a zero-length tangent
    result = derive_headings(
        [(0, 0), (1, 0), (1, 0), (1, 0), (2, 0)], start_theta=0.0, goal_theta=0.0
    )
    assert [(x, y) for x, y, _ in result] == [(0, 0), (1, 0), (2, 0)]
    assert all(math.isfinite(theta) for _, _, theta in result)


def test_nearest_index_exact_match():
    path = [(0, 0), (1, 0), (2, 0), (3, 0)]
    assert nearest_index((2, 0), path) == 2


def test_nearest_index_closest_not_exact():
    path = [(0, 0), (1, 0), (2, 0), (3, 0)]
    assert nearest_index((2.4, 0.1), path) == 2


def test_nearest_index_empty_path_does_not_crash():
    assert nearest_index((0, 0), []) == 0
