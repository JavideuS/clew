"""HTTP client for Spooky's global multi-robot planner.

One POST /v1/plan call per planning cycle, carrying every robot's spec in a
single request so Spooky can solve the joint QUBO/quantum-classical
formulation across the whole fleet at once.
FastAPI is the mantained integration surface for Spooky.

    POST {base_url}/v1/plan
    {
      "map_id": ..., "solver": ..., "format": "grid"|"graph",
      "robots": [{"id", "start": [x, y], "goal": [x, y], "start_time",
                  "priority", "robot_radius", "inflation",
                  "coordinate_format"}, ...],
      "penalty_set": ..., "clip_at_goal": true, "clearance_enabled": true
    }
    ->
    {
      "paths": [{"robot_id", "path": [[x, y], ...], "coordinate_format"}, ...],
      "cost": <float>
    }

Note for ordering.py: consecutive *duplicate* points in a returned path are
NOT deduplicated here. Here,
that same repetition is exactly the ordering signal release-gating cares
about ("this robot is meant to wait here"), so it must survive intact into
ordering.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import requests

from .robot import Fleet


class SpookyPlanError(RuntimeError):
    """Raised for any failure calling or parsing Spooky's /v1/plan."""


@dataclass
class SpookySettings:
    """Configuration for Spooky server.

    base_url: Server url
    solver_name: Solver name -- must be a key GET /solvers actually
        registers (e.g. "classic.cbs", "classic.ilp", "dwave.fast",
        "pennylane.inference.qaoa_CPU_fast", ...). Defaulted to
        "classic.cbs" instead: classical, exact, no GPU/quantum-hardware
        credentials needed, and fast.
    map_id: Map id -- must be a key GET /v1/maps actually lists. "default"
        -- the previous default here -- isn't registered on a live server
        either; defaulted to "no_obs5x5"
    penalty_set: Penalty set
    format: Format
    timeout_s: Timeout in seconds
    clip_at_goal: Clip at goal flag
    clearance_enabled: When true, Spooky expands obstacles (and other
        robots' reservations) by each robot's robot_radius + inflation
        before solving, so the returned paths already respect the real
        inflated footprint -- matches how the robots actually execute, and
        keeps release-gating (ordering.py, same combined effective radius)
        from having to serialise conflicts Spooky could have routed around.
        Turn off only to fall back to Spooky's centre-point-only routing.
    """

    base_url: str = "http://localhost:8000"
    solver_name: str = "classic.cbs"
    map_id: str = "no_obs5x5"
    penalty_set: str = "crash"
    format: str = "grid"  # "grid" or "graph"
    timeout_s: float = 30.0
    clip_at_goal: bool = True
    clearance_enabled: bool = True


@dataclass
class RobotPlan:
    """One robot's raw path as returned by Spooky.
    Symbolic-timestep order is preserved.
    """

    robot_id: str
    path: list[tuple[float, float]]
    coordinate_format: str


@dataclass
class FleetPlan:
    robot_plans: dict[str, RobotPlan]
    cost: float


def plan_fleet(fleet: Fleet, settings: SpookySettings) -> FleetPlan:
    """Call Spooky's POST /v1/plan once for the whole fleet.

    Raises SpookyPlanError on any request/response failure.
    Never returns a partial/best-effort plan.
    """
    if not len(fleet):
        raise SpookyPlanError("plan_fleet: fleet has no robots")

    body = {
        "map_id": settings.map_id,
        "solver": settings.solver_name,
        "format": settings.format,
        "robots": [robot.to_spooky_spec() for robot in fleet],
        "penalty_set": settings.penalty_set,
        "clip_at_goal": settings.clip_at_goal,
        "clearance_enabled": settings.clearance_enabled,
    }

    url = f"{settings.base_url}/v1/plan"
    try:
        resp = requests.post(url, json=body, timeout=settings.timeout_s)
    except requests.RequestException as exc:
        raise SpookyPlanError(f"spooky: POST {url} failed: {exc}") from exc

    if resp.status_code != 200:
        raise SpookyPlanError(
            f"spooky: server returned HTTP {resp.status_code}: {resp.text}"
        )

    try:
        data = resp.json()
    except ValueError as exc:
        raise SpookyPlanError(
            f"spooky: could not decode response as JSON: {exc}"
        ) from exc

    paths = data.get("paths")
    if not paths:
        raise SpookyPlanError("spooky: response contained no robot paths")

    robot_plans: dict[str, RobotPlan] = {}
    for entry in paths:
        robot_id = entry.get("robot_id")
        robot = fleet.robots.get(robot_id)
        if robot is None:
            raise SpookyPlanError(
                f"spooky: response contained unknown robot id {robot_id!r}"
            )

        coord_format = entry.get("coordinate_format")
        expected = robot.coordinate_format.value
        if coord_format != expected:
            raise SpookyPlanError(
                f"spooky: requested {expected!r} path for robot {robot_id!r}, "
                f"got {coord_format!r} -- check server version"
            )

        raw_path = entry.get("path") or []
        if len(raw_path) < 2:
            raise SpookyPlanError(
                f"spooky: robot {robot_id!r} path has {len(raw_path)} "
                "point(s), need >=2"
            )

        path = [(point[0], point[1]) for point in raw_path]

        # Pin the exact mission start/goal onto the first/last point
        # ("avoids grid-rounding drift"). Spooky quantizes world-format
        # start/goal to the nearest grid cell center before solving
        # (world_to_grid_cell), so the returned path's endpoints are only
        # ever *approximately* where the robot was actually asked to go --
        # left unpinned, the robot converges on the quantized cell center
        # instead of the real goal
        #
        # For "cartesian"/"matrix" the returned points are grid-cell indices;
        # pinning a world-metre Pose2D onto those would silently mix units.
        # Not implemented for those formats -- "world" is this client's
        # only fully-supported path anyway (see robot.py).
        if coord_format == "world":
            path[0] = (robot.start.x, robot.start.y)
            path[-1] = (robot.goal.x, robot.goal.y)

        robot_plans[robot_id] = RobotPlan(
            robot_id=robot_id,
            path=path,
            coordinate_format=coord_format,
        )

    if len(robot_plans) != len(fleet):
        missing = set(fleet.robots) - set(robot_plans)
        raise SpookyPlanError(
            f"spooky: requested {len(fleet)} robot paths, got "
            f"{len(robot_plans)} (missing: {sorted(missing)})"
        )

    return FleetPlan(robot_plans=robot_plans, cost=data.get("cost", 0.0))
