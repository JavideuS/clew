"""Pure path/pose helpers for dispatch.py

Pure math only (no ROS dependency at all)
"""

from __future__ import annotations

from .geometry import Point, Pose, distance, heading_to


def dedupe_consecutive(points: list[Point]) -> list[Point]:
    deduped: list[Point] = []
    for point in points:
        if not deduped or deduped[-1] != point:
            deduped.append(point)
    return deduped


def derive_headings(
    points: list[Point],
    start_theta: float,
    goal_theta: float | None,
) -> list[Pose]:
    """Dedupe consecutive duplicate points (a robot "waiting", per
    ordering.py's own docstring) and derive a heading per point via
    tangent-to-next, pinning the first point to start_theta. Left
    un-deduped, a repeated point produces a zero-length tangent, so this
    collapses them first.

    goal_theta: pin the LAST point's heading to this value -- pass the
    robot's real mission goal_theta ONLY when `points` is genuinely the
    robot's complete, final path (released_path() has caught up to the
    full path length). Pass None for a partial/intermediate release: with
    the mission goal_theta pinned unconditionally, every partial release's
    last point (just wherever release-gating currently allows, not the
    robot's actual destination) looks like a real arrival to the
    controller, which then rotates to face the far-off *mission* goal at
    every gated pause instead of only on genuine arrival. When None, the
    last point gets the same backward-tangent heading as any other
    interior point instead of being pinned to anything.

    Returns [] for an empty input. A single-point result (the whole
    released prefix collapsed to one point) gets start_theta regardless of
    goal_theta -- released_path() always starts at index 0, so a
    single-point result means "barely started," never "arrived," even
    though positionally it could be either.
    """
    deduped = dedupe_consecutive(points)
    if not deduped:
        return []
    if len(deduped) == 1:
        x, y = deduped[0]
        return [(x, y, start_theta)]

    result: list[Pose] = []
    for i, (x, y) in enumerate(deduped):
        if i == 0:
            theta = start_theta
        elif i == len(deduped) - 1 and goal_theta is not None:
            theta = goal_theta
        else:
            theta = heading_to(deduped[i - 1], (x, y))
        result.append((x, y, theta))
    return result


def nearest_index(point: Point, path: list[Point]) -> int:
    """Index of the closest point in `path` to `point`. O(len(path)) linear
    scan (fine for realistic path lengths).

    Safe to call with a slightly-stale/noisy live pose even if it lands
    nearer an earlier point than the robot's true progress:
    ReleaseSchedule.report_progress() treats reports as monotonic (max with
    the existing cleared index), so an occasional spurious backward result
    is harmless, not a correctness bug.
    """
    best_idx, best_dist = 0, float("inf")
    for i, candidate in enumerate(path):
        dist = distance(point, candidate)
        if dist < best_dist:
            best_dist, best_idx = dist, i
    return best_idx
