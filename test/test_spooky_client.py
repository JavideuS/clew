"""Unit tests for spooky_client.py -- global planning, HTTP mocked out.

None of these hit a real Spooky server -- see test_spooky_client_live.py
for that. Run with: pytest test/test_spooky_client.py
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from clew.robot import CoordinateFormat, Fleet
from clew.spooky_client import SpookyPlanError, SpookySettings, plan_fleet


@pytest.fixture
def two_robot_fleet() -> Fleet:
    return Fleet.from_specs(
        [
            {"id": "ranger_1", "start": {"x": 0.0, "y": 0.0}, "goal": {"x": 4.5, "y": 2.0}},
            {"id": "ranger_2", "start": {"x": 1.0, "y": 0.0}, "goal": {"x": 4.5, "y": -2.0}},
        ]
    )


def _mock_post(response_json: dict, status_code: int = 200):
    resp = MagicMock(status_code=status_code, text=str(response_json))
    resp.json.return_value = response_json
    return patch("clew.spooky_client.requests.post", return_value=resp)


def test_plan_fleet_empty_fleet_raises():
    with pytest.raises(SpookyPlanError, match="no robots"):
        plan_fleet(Fleet(), SpookySettings())


def test_plan_fleet_success(two_robot_fleet):
    response = {
        "paths": [
            {"robot_id": "ranger_1", "path": [[0, 0], [1, 1], [4.5, 2.0]], "coordinate_format": "world"},
            {"robot_id": "ranger_2", "path": [[1, 0], [4.5, -2.0]], "coordinate_format": "world"},
        ],
        "cost": 12.5,
    }
    with _mock_post(response):
        plan = plan_fleet(two_robot_fleet, SpookySettings())
    assert plan.cost == 12.5
    assert plan.robot_plans["ranger_1"].path == [(0, 0), (1, 1), (4.5, 2.0)]


def test_plan_fleet_sends_clearance_enabled(two_robot_fleet):
    response = {
        "paths": [
            {"robot_id": "ranger_1", "path": [[0, 0], [4.5, 2.0]], "coordinate_format": "world"},
            {"robot_id": "ranger_2", "path": [[1, 0], [4.5, -2.0]], "coordinate_format": "world"},
        ],
        "cost": 1.0,
    }
    with _mock_post(response) as post:
        plan_fleet(two_robot_fleet, SpookySettings())
        default_body = post.call_args.kwargs["json"]
    assert default_body["clearance_enabled"] is True

    with _mock_post(response) as post:
        plan_fleet(two_robot_fleet, SpookySettings(clearance_enabled=False))
        off_body = post.call_args.kwargs["json"]
    assert off_body["clearance_enabled"] is False


def test_plan_fleet_preserves_consecutive_duplicates(two_robot_fleet):
    # a robot "waiting" at a cell -- must survive intact for ordering.py,
    # which relies on the repetition as its "wait here" signal.
    response = {
        "paths": [
            {"robot_id": "ranger_1", "path": [[0, 0], [1, 0], [1, 0], [4.5, 2.0]], "coordinate_format": "world"},
            {"robot_id": "ranger_2", "path": [[1, 0], [4.5, -2.0]], "coordinate_format": "world"},
        ],
        "cost": 4.0,
    }
    with _mock_post(response):
        plan = plan_fleet(two_robot_fleet, SpookySettings())
    assert plan.robot_plans["ranger_1"].path == [(0, 0), (1, 0), (1, 0), (4.5, 2.0)]


def test_plan_fleet_http_error_status(two_robot_fleet):
    with _mock_post({}, status_code=500):
        with pytest.raises(SpookyPlanError, match="HTTP 500"):
            plan_fleet(two_robot_fleet, SpookySettings())


def test_plan_fleet_connection_failure(two_robot_fleet):
    import requests

    with patch(
        "clew.spooky_client.requests.post",
        side_effect=requests.ConnectionError("refused"),
    ):
        with pytest.raises(SpookyPlanError, match="POST .* failed"):
            plan_fleet(two_robot_fleet, SpookySettings())


def test_plan_fleet_no_paths_in_response(two_robot_fleet):
    with _mock_post({"paths": [], "cost": 0.0}):
        with pytest.raises(SpookyPlanError, match="no robot paths"):
            plan_fleet(two_robot_fleet, SpookySettings())


def test_plan_fleet_unknown_robot_id_in_response(two_robot_fleet):
    response = {
        "paths": [{"robot_id": "ghost", "path": [[0, 0], [1, 1]], "coordinate_format": "world"}],
        "cost": 1.0,
    }
    with _mock_post(response):
        with pytest.raises(SpookyPlanError, match="unknown robot id"):
            plan_fleet(two_robot_fleet, SpookySettings())


def test_plan_fleet_coordinate_format_mismatch(two_robot_fleet):
    response = {
        "paths": [
            {"robot_id": "ranger_1", "path": [[0, 0], [1, 1]], "coordinate_format": "cartesian"},
        ],
        "cost": 1.0,
    }
    with _mock_post(response):
        with pytest.raises(SpookyPlanError, match="check server version"):
            plan_fleet(two_robot_fleet, SpookySettings())


def test_plan_fleet_path_too_short(two_robot_fleet):
    response = {
        "paths": [
            {"robot_id": "ranger_1", "path": [[0, 0]], "coordinate_format": "world"},
        ],
        "cost": 1.0,
    }
    with _mock_post(response):
        with pytest.raises(SpookyPlanError, match="need >=2"):
            plan_fleet(two_robot_fleet, SpookySettings())


def test_plan_fleet_pins_grid_quantized_endpoints_to_exact_mission_goal(two_robot_fleet):
    # Spooky quantizes world-format start/goal to the nearest grid cell
    # center -- ranger_1's real goal is (4.5, 2.0), but the server's grid
    # only landed near it, at (4.475, 1.975). Left unpinned, the robot
    # would converge on the quantized cell center, not the real goal (this
    # is exactly what a real sim run surfaced: requesting (1.2, 1.2) on a
    # 0.4 m grid, the robot settled around (0.99, 0.94) instead).
    response = {
        "paths": [
            {
                "robot_id": "ranger_1",
                "path": [[0.025, 0.025], [1.0, 1.0], [4.475, 1.975]],
                "coordinate_format": "world",
            },
            {"robot_id": "ranger_2", "path": [[1, 0], [4.5, -2.0]], "coordinate_format": "world"},
        ],
        "cost": 12.5,
    }
    with _mock_post(response):
        plan = plan_fleet(two_robot_fleet, SpookySettings())
    path = plan.robot_plans["ranger_1"].path
    assert path[0] == (0.0, 0.0)  # ranger_1's exact Robot.start, not (0.025, 0.025)
    assert path[-1] == (4.5, 2.0)  # ranger_1's exact Robot.goal, not (4.475, 1.975)
    assert path[1] == (1.0, 1.0)  # interior points untouched


def test_plan_fleet_does_not_pin_non_world_formats(two_robot_fleet):
    # cartesian/matrix path points are grid-cell indices, not metres --
    # pinning a world-metre Robot.start/goal onto them would mix units.
    two_robot_fleet.robots["ranger_1"].coordinate_format = CoordinateFormat.CARTESIAN
    response = {
        "paths": [
            {"robot_id": "ranger_1", "path": [[0, 0], [3, 4]], "coordinate_format": "cartesian"},
            {"robot_id": "ranger_2", "path": [[1, 0], [4.5, -2.0]], "coordinate_format": "world"},
        ],
        "cost": 1.0,
    }
    with _mock_post(response):
        plan = plan_fleet(two_robot_fleet, SpookySettings())
    assert plan.robot_plans["ranger_1"].path == [(0, 0), (3, 4)]


def test_plan_fleet_missing_robot_in_response(two_robot_fleet):
    response = {
        "paths": [
            {"robot_id": "ranger_1", "path": [[0, 0], [1, 1]], "coordinate_format": "world"},
        ],
        "cost": 1.0,
    }
    with _mock_post(response):
        with pytest.raises(SpookyPlanError, match="missing"):
            plan_fleet(two_robot_fleet, SpookySettings())
