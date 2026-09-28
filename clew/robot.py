"""Mission/goal intake: the Robot class and the Fleet container.

Each robot carries its own independent parameters
Deliberately mirroring Spooky's own wire format (`RobotSpec`: id, start, goal, start_time,
priority, robot_radius, inflation, coordinate_format -- see spooky_client.py).

A `Robot` carries a full `Pose2D` (with heading) for its start/goal, not
just an (x, y) pair.
Spooky itself never sees `theta` but the local planner downstream needs a real
target heading, so the mission's exact start/goal poses are pinned back in
after Spooky's response rather than trusting a grid-derived heading.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import yaml


class CoordinateFormat(str, Enum):
    """
    WORLD (default, recommended): real-world (x, y) metres, sent/received as-is, no grid-cell quantization.
    CARTESIAN/MATRIX: grid-cell indices, kept only for servers/maps where "world" isn't available.
    """

    WORLD = "world"
    CARTESIAN = "cartesian"
    MATRIX = "matrix"


@dataclass(frozen=True)
class Pose2D:
    """A world-space pose. theta is radians, purely a local-planner concern."""

    x: float
    y: float
    theta: float = 0.0


@dataclass
class Robot:
    """One robot's independent planning parameters.

    Fields map 1:1 onto Spooky's RobotSpec (see spooky_client.py) except
    `start`/`goal`, which carry full Pose2D here (theta included) and are
    reduced to bare (x, y) only when serialized for Spooky.
    """

    id: str
    start: Pose2D
    goal: Pose2D
    start_time: int = 0
    priority: float = 1.0
    robot_radius: float = 0.35
    inflation: float = 0.0
    coordinate_format: CoordinateFormat = CoordinateFormat.WORLD

    @property
    def effective_radius(self) -> float:
        """`robot_radius + inflation` -- how far another robot's centre must
        stay outside this one for the two footprints (plus local-costmap
        inflation) to actually clear each other in execution.

        Spooky gets `robot_radius` and `inflation` as separate fields (it
        inflates obstacles by their sum itself); this bundled value is what
        the downstream execution-safety layers compare against -- release-
        gating (`ordering.py`) and the deadlock-proximity check
        (`coordinator_node`).
        """
        return self.robot_radius + self.inflation

    def to_spooky_spec(self) -> dict:
        """Serialize to Spooky's RobotSpec JSON shape.

        Theta is intentionally dropped here, Spooky's solver works over (x, y) positions only;
        heading is resolved locally once a path comes back, pinning the mission's exact start/goal Pose2D
        onto the first/last path point.
        """
        return {
            "id": self.id,
            "start": [self.start.x, self.start.y],
            "goal": [self.goal.x, self.goal.y],
            "start_time": self.start_time,
            "priority": self.priority,
            "robot_radius": self.robot_radius,
            "inflation": self.inflation,
            "coordinate_format": self.coordinate_format.value,
        }


@dataclass
class Fleet:
    """The full set of robots for one planning cycle."""

    robots: dict[str, Robot] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for robot_id, robot in self.robots.items():
            if robot_id != robot.id:
                raise ValueError(
                    f"fleet: robot keyed as {robot_id!r} but Robot.id is {robot.id!r}"
                )

    @classmethod
    def from_specs(cls, specs: list[dict]) -> Fleet:
        """Build a Fleet from a list of plain dicts, e.g. loaded from a mission YAML/JSON file.
        Each dict's keys match Robot's fields; `start`/`goal` are {x, y, theta} (theta optional, default 0.0).

        Example:
            [{"id": "ranger_1",
              "start": {"x": 0.0, "y": 0.0},
              "goal":  {"x": 4.5, "y": 2.0, "theta": 1.57},
              "robot_radius": 0.6,
              "inflation": 0.2,
              "priority": 1.0,
              }
              ]
        """
        robots: dict[str, Robot] = {}
        for spec in specs:
            spec = dict(spec)
            robot_id = spec.pop("id")
            start = Pose2D(**spec.pop("start"))
            goal = Pose2D(**spec.pop("goal"))
            coordinate_format = CoordinateFormat(
                spec.pop("coordinate_format", CoordinateFormat.WORLD.value)
            )
            robots[robot_id] = Robot(
                id=robot_id,
                start=start,
                goal=goal,
                coordinate_format=coordinate_format,
                **spec,
            )
        return cls(robots=robots)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Fleet:
        """Load a mission file: a YAML list of robot specs, same shape as
        from_specs (see config/mission.example.yaml for a worked example).
        """
        with open(path) as f:
            specs = yaml.safe_load(f)
        if not isinstance(specs, list):
            raise ValueError(
                f"fleet: mission file {path!s} must contain a YAML list of "
                f"robot specs, got {type(specs).__name__}"
            )
        return cls.from_specs(specs)

    def __len__(self) -> int:
        return len(self.robots)

    def __iter__(self):
        return iter(self.robots.values())
