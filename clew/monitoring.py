"""Pure decision logic for coordinator_node's three execution-health
checks -- fleet-stall detection, fleet-deadlock detection, and initial
localization convergence. No ROS dependency: each check's actual data
source (a live pose, a TF lookup, which dispatchers report stuck) stays a
method on CoordinatorNode; what that data *means* -- update a streak, is
this a stall, is this systematic -- lives here instead, as functions
taking their config and inputs explicitly. ROS-free, unit-testable
without rclpy (see test_monitoring.py).

Config dataclasses here mirror spooky_client.SpookySettings: each field's
default is also a coordinator_node.py declare_parameter default, kept in
sync by hand since ROS2 parameters can't reference a dataclass directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .geometry import Point, angle_diff, distance

# (anchor_pos, anchor_released_len, consecutive ticks with no movement/
# progress since the anchor was set) -- see StallConfig / stall_anchor_update.
StallAnchor = tuple[Point, int, int]


@dataclass(frozen=True)
class StallConfig:
    """Fleet-stall detection: >=2 not-yet-home robots with no movement and
    no newly released path for `ticks` dispatch ticks in a row, with at
    least one release-gated, means the fleet is stopped clean at a mutual
    gate (no controller abort ever fires) -- hand to recovery. `ticks`
    long enough not to trip on a normal single-robot gated wait while its
    blocker moves.
    """

    ticks: int = 16
    pos_eps_m: float = 0.15  # movement below this since the anchor was set = stopped


def stall_anchor_update(
    anchor: StallAnchor | None,
    pos: Point,
    released: int,
    full_path: list[Point],
    cfg: StallConfig,
) -> tuple[StallAnchor | None, bool]:
    """One dispatch tick's stall-anchor update for a single robot.
    Returns (new_anchor, is_stalled).

    new_anchor is None once the robot is at its goal (released its whole
    path and within cfg.pos_eps_m of its end) -- nothing left to track.
    Otherwise: any real displacement from the anchor (>= cfg.pos_eps_m) or
    newly released path re-anchors here and restarts the streak at 0 -- a
    fixed anchor rather than a trailing window, so a single AMCL jitter
    spike costs at most one reset instead of poisoning a whole window's
    worth of ticks. is_stalled goes True once the streak against an
    unmoved anchor reaches cfg.ticks.
    """
    at_goal = (
        released >= len(full_path) and distance(pos, full_path[-1]) < cfg.pos_eps_m
    )
    if at_goal:
        return None, False

    if (
        anchor is None
        or distance(pos, anchor[0]) >= cfg.pos_eps_m
        or released > anchor[1]
    ):
        return (pos, released, 0), False

    anchor_pos, anchor_released, streak = anchor
    streak += 1
    return (anchor_pos, anchor_released, streak), streak >= cfg.ticks


@dataclass(frozen=True)
class DeadlockConfig:
    """Two stuck robots closer than `proximity_factor` times their
    combined effective radius (robot_radius + inflation) count as
    mutually blocking (a little slack over 1.0, since real footprints
    stop short of contact).
    """

    proximity_factor: float = 1.25


def deadlock_group(
    stuck_ids: list[str],
    positions: dict[str, Point],
    effective_radii: dict[str, float],
    pending_blockers: dict[str, list[tuple[str, int, int]]],
    cfg: DeadlockConfig,
) -> tuple[
    frozenset[str], list[tuple[str, str, float, float]], list[tuple[str, str, int, int]]
]:
    """(group, collisions, chains) for the currently-stuck robots.

    `collisions`: pairs of stuck robots within cfg.proximity_factor *
    their combined effective radius. `chains`: (robot, stuck_blocker,
    need, have) for any robot release-gated behind a stuck one --
    `pending_blockers` covers every robot, stuck or not, so a chain can
    reach a robot that isn't itself stuck. `group` is every robot id named
    in either; empty (with empty collisions/chains) if nothing qualifies.
    """
    collisions: list[tuple[str, str, float, float]] = []
    for i, a in enumerate(stuck_ids):
        for b in stuck_ids[i + 1 :]:
            gap = distance(positions[a], positions[b])
            limit = effective_radii[a] + effective_radii[b]
            if gap <= limit * cfg.proximity_factor:
                collisions.append((a, b, gap, limit))

    stuck_set = set(stuck_ids)
    chains: list[tuple[str, str, int, int]] = []
    for rid, blockers in pending_blockers.items():
        for other, need, have in blockers:
            if other in stuck_set:
                chains.append((rid, other, need, have))

    group = frozenset(
        {rid for pair in collisions for rid in pair[:2]}
        | {rid for chain in chains for rid in chain[:2]}
    )
    return group, collisions, chains


@dataclass(frozen=True)
class ConvergenceConfig:
    """Initial-localization convergence gate: every robot must converge on
    its seeded start -- checked in TF (what the controller actually
    reads; amcl_pose alone isn't reliable enough) -- before planning
    starts, all-or-nothing across the fleet (a partial start shifts every
    cross-robot deadline in ordering.py off its symbolic-step pacing).
    """

    poll_period_s: float = 0.5
    pos_tol_m: float = 0.35
    yaw_tol_rad: float = 0.25
    # Real-hardware gate: converged means map->base_frame has *stopped
    # moving* for stable_polls in a row, not that it matches the declared
    # seed (see ConvergenceGate's docstring -- on real hardware the
    # declared start is a kick-start hint, not a promise, and AMCL
    # scan-matching a couple meters away from an inaccurate one is
    # correct behavior, not a bug). These two are deliberately tighter
    # than pos_tol_m/yaw_tol_rad, which stay in use for the declared-vs-
    # actual diagnostic in _report_offset_diagnostics.
    settle_pos_tol_m: float = 0.05
    settle_yaw_tol_rad: float = 0.05
    stable_polls: int = 4  # must hold within tolerance this many polls in a row
    # Once settled, if it's still farther than pos_tol_m/yaw_tol_rad from
    # the declared start, re-seed at start - (settled - start) -- assumes
    # AMCL's local pull is roughly constant nearby, so aiming this far
    # past the declared point, opposite the observed drift, lands closer
    # to it next time -- and re-settle, up to this many times. The
    # closest-to-declared settle seen across all rounds is what's used
    # even if none ever lands inside tolerance (0 disables: first settle
    # wins, whatever the offset).
    max_correction_rounds: int = 3
    timeout_s: float = 90.0  # backstop: past this, plan anyway but loudly
    log_every_n_polls: int = 4  # throttle the "still not converged" line
    offset_history_len: int = (
        12  # recent (TF - seed) samples kept, for the timeout report
    )
    # Re-seed cadence while unconverged: a burst up front (a publish on a
    # just-created publisher can miss its subscriber), then spaced out so
    # re-seeding doesn't outrun AMCL's own filter updates.
    reseed_burst_polls: int = 2
    reseed_every_n_polls: int = 6
    # How tightly every robot's mean (TF - seed) offset must agree (in
    # both x and y) to call a shared offset "systematic" rather than
    # per-robot noise -- see systematic_offset.
    systematic_agreement_tol_m: float = 0.30


def convergence_offset(
    tf_pos: Point,
    tf_yaw: float,
    seed: tuple[float, float, float],
    cfg: ConvergenceConfig,
) -> tuple[float, float, float, str | None]:
    """(dx, dy, dyaw, blocker) for one convergence poll's TF sample against
    its seeded start `seed` = (x, y, theta). blocker is None once within
    tolerance, else a short string naming what's still off.
    """
    seed_x, seed_y, seed_theta = seed
    dx, dy = tf_pos[0] - seed_x, tf_pos[1] - seed_y
    dist = distance(tf_pos, (seed_x, seed_y))
    dyaw = angle_diff(tf_yaw, seed_theta)

    if dist > cfg.pos_tol_m:
        return (
            dx,
            dy,
            dyaw,
            f"TF puts it at ({tf_pos[0]:.2f}, {tf_pos[1]:.2f}), {dist:.2f} m "
            f"from the seeded start ({seed_x}, {seed_y}) (tol {cfg.pos_tol_m} m)",
        )
    if abs(dyaw) > cfg.yaw_tol_rad:
        return (
            dx,
            dy,
            dyaw,
            f"TF yaw {tf_yaw:.2f} is {abs(dyaw):.2f} rad from the seeded "
            f"{seed_theta} (tol {cfg.yaw_tol_rad})",
        )
    return dx, dy, dyaw, None


def convergence_streak_update(
    streak: int, blocked: bool, cfg: ConvergenceConfig
) -> tuple[int, bool]:
    """New streak count and whether this poll just reached
    cfg.stable_polls (this robot is now converged). blocked=True resets
    the streak to 0; otherwise it increments.
    """
    if blocked:
        return 0, False
    new_streak = streak + 1
    return new_streak, new_streak >= cfg.stable_polls


def systematic_offset(means: dict[str, Point], cfg: ConvergenceConfig) -> bool:
    """True if every robot's mean (TF - seed) offset agrees within
    cfg.systematic_agreement_tol_m (both x and y) AND the shared offset
    itself exceeds cfg.pos_tol_m -- i.e. a systematic frame error (map
    origin, sensor mount offset) rather than per-robot localization
    noise. Needs >= 2 robots to mean anything; False for fewer.
    """
    if len(means) < 2:
        return False
    xs = [m[0] for m in means.values()]
    ys = [m[1] for m in means.values()]
    agree = (max(xs) - min(xs)) < cfg.systematic_agreement_tol_m and (
        max(ys) - min(ys)
    ) < cfg.systematic_agreement_tol_m
    mag = math.hypot(sum(xs) / len(xs), sum(ys) / len(ys))
    return agree and mag > cfg.pos_tol_m
