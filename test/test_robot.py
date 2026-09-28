"""Unit tests for robot.py -- mission/goal intake.

Run with: pytest test/test_robot.py  (or just `pytest`, from the repo root)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from clew.robot import CoordinateFormat, Fleet, Pose2D, Robot

MISSION_EXAMPLE = Path(__file__).parent.parent / "config" / "mission.example.yaml"


def test_robot_to_spooky_spec_drops_theta():
    robot = Robot(
        id="ranger_1",
        start=Pose2D(x=0.0, y=0.0, theta=0.3),
        goal=Pose2D(x=4.5, y=2.0, theta=1.57),
        robot_radius=0.6,
        inflation=0.2,
    )
    spec = robot.to_spooky_spec()
    assert spec == {
        "id": "ranger_1",
        "start": [0.0, 0.0],
        "goal": [4.5, 2.0],
        "start_time": 0,
        "priority": 1.0,
        "robot_radius": 0.6,
        "inflation": 0.2,
        "coordinate_format": "world",
    }


def test_robot_defaults():
    robot = Robot(id="r", start=Pose2D(0, 0), goal=Pose2D(1, 1))
    assert robot.start_time == 0
    assert robot.priority == 1.0
    assert robot.robot_radius == 0.35
    assert robot.inflation == 0.0
    assert robot.coordinate_format == CoordinateFormat.WORLD


def test_robot_effective_radius_sums_radius_and_inflation():
    robot = Robot(
        id="r", start=Pose2D(0, 0), goal=Pose2D(1, 1),
        robot_radius=0.6, inflation=0.2,
    )
    assert robot.effective_radius == pytest.approx(0.8)


def test_fleet_from_specs_builds_robots():
    fleet = Fleet.from_specs(
        [
            {
                "id": "ranger_1",
                "start": {"x": 0.0, "y": 0.0},
                "goal": {"x": 4.5, "y": 2.0, "theta": 1.57},
                "robot_radius": 0.6,
                "inflation": 0.2,
            },
            {
                "id": "ranger_2",
                "start": {"x": 1.0, "y": 0.0},
                "goal": {"x": 4.5, "y": -2.0},
            },
        ]
    )
    assert len(fleet) == 2
    assert fleet.robots["ranger_1"].goal == Pose2D(4.5, 2.0, 1.57)
    assert fleet.robots["ranger_2"].goal.theta == 0.0  # default
    assert {r.id for r in fleet} == {"ranger_1", "ranger_2"}


def test_fleet_from_specs_supports_non_default_coordinate_format():
    fleet = Fleet.from_specs(
        [
            {
                "id": "r",
                "start": {"x": 0, "y": 0},
                "goal": {"x": 4, "y": 4},
                "coordinate_format": "cartesian",
            }
        ]
    )
    assert fleet.robots["r"].coordinate_format == CoordinateFormat.CARTESIAN


def test_fleet_rejects_mismatched_keys():
    with pytest.raises(ValueError, match="Robot.id"):
        Fleet(robots={"wrong_key": Robot(id="right_id", start=Pose2D(0, 0), goal=Pose2D(1, 1))})


def test_fleet_from_yaml_loads_example_mission():
    fleet = Fleet.from_yaml(MISSION_EXAMPLE)
    assert len(fleet) == 2
    assert set(fleet.robots) == {"ranger_1", "ranger_2"}
    assert fleet.robots["ranger_1"].robot_radius == 0.35


def test_fleet_from_yaml_rejects_non_list(tmp_path):
    bad_file = tmp_path / "bad.yaml"
    bad_file.write_text("not_a_list: true\n")
    with pytest.raises(ValueError, match="must contain a YAML list"):
        Fleet.from_yaml(bad_file)
