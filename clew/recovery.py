"""Coordinated deadlock recovery -- the fleet's emergency fallback for when
the global plan, *as executed*, wedges two or more robots against each
other with no forward progress possible. Release-gating (ordering.py) can't
resolve that on its own: the gate is behaving correctly, the plan itself
put the robots somewhere too tight to share.

Normally this never runs. It exists so that a planner producing a bad plan
degrades to "the fleet sorts it out and keeps going" rather than "everything stops". With a
clearance-aware planner the gate handles the sharing and recovery stays dormant.

Trigger: coordinator_node._check_fleet_deadlock / _check_fleet_stall spot a
mutual block (two stuck/stalled robots inside their combined effective
radius, or a robot gated behind one) and call RecoveryManager.on_deadlock.

Two recovery strategies, tried in this order per attempt:

  LIGHT (attempt 1 only, exactly two robots, a structural bottleneck
  between them -- ordering.py.ReleaseSchedule.structural_bottleneck):
    hold both. Drive the `second` robot (the one holding at the zone
    mouth) clear of `first`'s *entire remaining route*, not just the
    shared zone's points -- a structural bottleneck is specifically the
    case where `first`'s route stays close to `second` well past the zone
    itself (see ordering.py's own definition), so validating only against
    the zone can hand back a yield point that's still in `first`'s way
    further along, causing an immediate second deadlock once `first`
    resumes. Same yield ladder as the heavy path (whichever of
    retreat-along-its-own-trail or advance-along-its-own-route is the
    SHORTER real detour, else a Spooky sidestep) -- the move itself is
    still typically a short hop, only the obstacle set it's checked
    against is the wide one.
    If nothing clears the wide check, retry once against just the zone
    itself (the original, narrower set) before giving up on light
    recovery entirely: on a tight map, "first's route stays close to
    second for most of its remaining trip" (the structural definition
    itself) can mean no point anywhere is clear of the *whole* route, even
    though the immediate zone conflict has a perfectly fine answer --
    requiring wide clearance unconditionally can turn a recoverable
    conflict into a full RECOVERY FAILED needing an operator, which is
    worse than the reduced-confidence move the narrow retry falls back
    to (at worst, a second, later deadlock that gets caught and recovered
    same as any other). Once confirmed clear (wide or narrow):
    ReleaseSchedule.release_override(first, first_range) drops exactly the
    constraints ordering.py added for this bottleneck -- `first` resumes
    its OWN ALREADY-PLANNED path through the zone, no new plan. `second`'s
    own constraint (wait for `first`'s real progress past the zone) is
    untouched; it opens the normal way, off the SAME schedule, once
    `first` actually gets there, and `second` then resumes its own
    original path too -- FollowPath tracks from wherever a robot currently
    is, so driving away from the sidestepped position isn't a special case.
    No CBS call in the common case: the fix is "make the one thing that
    was physically false (second still being in the way) become true",
    not "compute a new plan neither robot needed changed".
    If `second` can't be confirmed clear against EITHER obstacle set (no
    yield pose found, or every yield attempt fails), falls through to
    HEAVY within the same attempt.

  HEAVY (the fallback, and every attempt for anything that isn't a clean
  2-robot structural pair): pick a winner -- the robot nearest its own
  goal (ties: higher priority, then lower id) -- hold the whole group, and
  yield each non-winner clear of the *winner's entire remaining route*
  (same ladder). Once every yielder that could be cleared is, replan the
  whole fleet from every robot's *current* pose and resume normal
  dispatch -- the gate then releases each yielder the instant the winner
  clears their shared zone, not at the winner's goal, so concurrency comes
  back on its own. Attempt 2 (the one retry) additionally pins the
  winner's priority high in the CBS request, forcing the solver to route
  the others around it -- a guaranteed path to goal for at least the
  winner.

  A one-off yield move that aborts is retried in place, up to
  RecoveryConfig.max_yield_retries times, before being given up on --
  replanning (or overriding) from a position recovery never actually
  confirmed clear just reproduces the same deadlock one tick later,
  burning a whole attempt on it for nothing.

Escalation: a rebuilt/overridden state that deadlocks again is attempt
N+1. Past RecoveryConfig.max_recovery_attempts: FAILED, fleet halted.

This module is ROS-free, purely logic-based.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from .geometry import Point, Pose, angle_diff, distance, heading_to
from .ordering import StructuralBottleneck


@dataclass(frozen=True)
class RecoveryConfig:
    """Tuning for coordinated deadlock recovery -- see this module's own
    docstring for the two-strategy (light/heavy) mechanism these numbers
    tune.
    """

    # Attempt 1 = the initial yield (light if applicable, else heavy) +
    # replan/override; attempt 2 = the one retry (heavy, winner pinned). A
    # third detection of the same group is FAILED.
    max_recovery_attempts: int = 2
    # Spooky HTTP timeout for a single recovery sidestep probe -- separate
    # from spooky.timeout_s: a probe is a small single-robot "is this cell
    # free?" query, not the full-fleet solve, and up to max_sidestep_probes
    # of these run serially per yielder -- at the full-fleet timeout, a
    # couple of slow/failed probes alone can burn tens of seconds of a
    # fleet already stopped waiting on recovery.
    sidestep_timeout_s: float = 5.0
    # Sideways-hop fallback: how many candidate staging goals to actually
    # try through Spooky (each a blocking single-robot plan call).
    # Candidates are ordered best-first -- perpendicular to the contested
    # corridor -- so a handful covers the useful directions without
    # stalling the fleet.
    max_sidestep_probes: int = 6
    # A one-off yield move (retreat / sidestep / advance) that aborts is
    # recomputed and retried from wherever the robot actually ended up, up
    # to this many times, before recovery gives up on clearing that robot
    # for the current attempt. Retrying the cheap yield beats spending a
    # whole replan/override cycle on a robot recovery never confirmed is
    # clear.
    max_yield_retries: int = 2
    # Priority handed to the winner on attempt 2 -- just has to dominate
    # every realistic mission priority so CBS resolves conflicts by moving
    # the others.
    pinned_winner_priority: float = 1_000_000.0


def _local_route_dir(route: list[Point]) -> Point:
    """A rough heading vector for the first stretch of `route` -- used to
    push a sideways hop perpendicular to the contested corridor. (0, 0) if
    the route is too short to have a direction."""
    if len(route) < 2:
        return (0.0, 0.0)
    ahead = route[min(3, len(route) - 1)]
    return (ahead[0] - route[0][0], ahead[1] - route[0][1])


# ─── Pure decision logic (no ROS -- see test_recovery.py) ───────────────────


def pick_winner(
    group: Iterable[str],
    remaining_counts: dict[str, int],
    priorities: dict[str, float],
) -> tuple[str, list[str]]:
    """(winner, [yielders]) for a deadlocked `group`. Winner = the robot
    nearest its own goal (fewest remaining path points); ties -> highest
    priority; ties -> lowest id. Everyone else yields.
    """
    ordered = sorted(
        group,
        key=lambda rid: (remaining_counts[rid], -priorities[rid], rid),
    )
    return ordered[0], ordered[1:]


def trail_retreat_point(
    trail: list[Point], obstacles: list[Point], clearance: float
) -> tuple[int, Point] | None:
    """The nearest point the robot can retreat to along `trail` (its
    already-driven points, start .. current): walking backward from just
    before its current position, the point at which retreating STOPS being
    safe -- i.e. the last point, closest to current, such that it AND
    every point between it and current clears `obstacles` by `clearance`.

    Checking the whole stretch it would actually drive back through (not
    just the retreat target in isolation) is essential, same reasoning as
    forward_advance_point: a target only reachable by first passing back
    through a closer conflict point is not a real yield. Because of that,
    this can only ever return the point immediately before the current
    one -- if THAT doesn't clear, nothing farther back does either, since
    reaching it means passing through the near point first. None if even
    that one step isn't clear, or the robot never moved (a single-point
    trail has nowhere to retreat to).
    """
    if len(trail) < 2:
        return None
    idx = len(trail) - 2
    if all(distance(trail[idx], o) >= clearance for o in obstacles):
        return idx, trail[idx]
    return None


def forward_advance_point(
    remaining: list[Point], obstacles: list[Point], clearance: float
) -> tuple[int, Point] | None:
    """Walk the yielder's *own remaining planned path* (index 0 = its
    current position) forward; return (index, point) of the furthest point
    reachable while EVERY step of the prefix stays at least `clearance`
    from every point in `obstacles`. None if even the first step isn't
    clear -- the yielder's route heads straight into the contested space
    and it can't advance out of the way along it.

    This is the "move a few steps along your own route to get clear" case
    -- for a robot with nothing on its trail to retreat along but whose
    own route genuinely leads away from the winner. Checking the whole
    prefix (not just the endpoint) is essential: a clear endpoint reached
    by driving back through the winner is not a yield.
    """
    best: tuple[int, Point] | None = None
    for idx in range(1, len(remaining)):
        if not all(distance(remaining[idx], o) >= clearance for o in obstacles):
            break
        best = (idx, remaining[idx])
    return best


def path_never_worsens(
    path: list[Point], obstacles: list[Point], clearance: float
) -> bool:
    """True if `path`'s distance to the nearest point in `obstacles` never
    drops below its own STARTING value as it goes (path[0] is wherever the
    robot currently is -- itself possibly still inside the conflict, so it
    can't be held to `clearance` yet), and the path actually finishes at
    or past `clearance`.

    This is what a Spooky-planned sidestep hop is checked against: not
    "every point clears `clearance` outright" (rejects any real gradual
    departure from a currently-blocked position -- the first interpolated
    step is barely different from the start, same mistake
    forward_advance_point's own docstring warns against for a robot's own
    route), and not "just the endpoint" either -- a route whose destination
    looks clear can still swing back near the conflict partway through and
    abort mid-transit. Monotonic non-worsening catches that dip while
    still accepting an ordinary steady departure.
    """
    if len(path) < 2:
        return False

    def min_dist(pt: Point) -> float:
        return min(distance(pt, o) for o in obstacles)

    baseline = min_dist(path[0])
    return (
        all(min_dist(p) >= baseline for p in path[1:])
        and min_dist(path[-1]) >= clearance
    )


def _forward_tangents(points: list[Point]) -> list[float]:
    """Heading at each point = direction toward the NEXT point (the last
    point inherits the previous heading). This is the body orientation a
    robot already has while driving forward along `points`.
    """
    out: list[float] = []
    for k in range(len(points)):
        if k + 1 < len(points):
            out.append(heading_to(points[k], points[k + 1]))
        elif out:
            out.append(out[-1])
        else:
            out.append(0.0)
    return out


def retreat_poses(trail: list[Point], retreat_idx: int) -> list[Pose]:
    """Poses to send the yielder back out to `trail[retreat_idx]`: the
    sub-trail from there to its current position, REVERSED so the current
    pose is first and the retreat point last, each heading kept at the
    original forward-travel tangent (not the reversed direction of motion).
    A controller that allows reversing follows this straight back; one that
    doesn't will rotate 180 once at the start.
    """
    seg = list(trail[retreat_idx:])  # retreat point .. current, forward order
    if len(seg) < 2:
        return []
    headings = _forward_tangents(seg)
    posed = [(seg[k][0], seg[k][1], headings[k]) for k in range(len(seg))]
    return posed[::-1]  # current first, retreat point last


def forward_poses(points: list[Point]) -> list[Pose]:
    """Bare points -> (x, y, theta) with forward-tangent headings, for a
    normally-driven one-off path (the sideways hop).
    """
    headings = _forward_tangents(points)
    return [(points[k][0], points[k][1], headings[k]) for k in range(len(points))]


def sideways_probe_goals(
    center: Point,
    route_dir: Point,
    step: float,
    count: int = 12,
    ring_factors: tuple[float, ...] = (1.2, 1.8, 2.6),
) -> list[Point]:
    """Candidate goals for a sideways hop: rings of `count` points around
    `center`, radius = ring_factor * step. Ordered so points roughly
    PERPENDICULAR to `route_dir` (the direction of the contested corridor
    at `center`) come first -- for a swap through one corridor the way out
    is to the side, not further along it. A near-zero `route_dir` falls
    back to an unbiased ring.
    """
    perp = math.atan2(route_dir[1], route_dir[0]) + math.pi / 2.0
    biased = math.hypot(route_dir[0], route_dir[1]) > 1e-6
    scored: list[tuple[float, Point]] = []
    for rf in ring_factors:
        r = rf * step
        for i in range(count):
            ang = 2.0 * math.pi * i / count
            # 0 when perpendicular to the corridor, pi/2 when along it
            off = abs(angle_diff(ang, perp)) if biased else 0.0
            off = min(off, math.pi - off)
            scored.append(
                (off, (center[0] + r * math.cos(ang), center[1] + r * math.sin(ang)))
            )
    scored.sort(key=lambda s: s[0])
    return [p for _, p in scored]


# ─── FSM ───────────────────────────────────────────────────────────────────


@dataclass
class RecoveryHooks:
    """Everything RecoveryManager needs from the coordinator. All ROS/IO
    lives behind these; the manager itself stays importable and testable
    without rclpy.
    """

    position_of: Callable[[str], Point]
    trail_of: Callable[[str], list[Point]]
    remaining_of: Callable[[str], list[Point]]
    # (robot_id, poses, on_done, reason) -- send a one-off FollowPath,
    # on_done(success: bool) when it terminates.
    drive_oneoff: Callable[[str, list[Pose], Callable[[bool], None], str], None]
    hold: Callable[[str], None]
    resume: Callable[[str], None]
    # The StructuralBottleneck between these two robots, if ordering.py
    # flagged one, else None.
    structural_bottleneck: Callable[[str, str], StructuralBottleneck | None]
    # Drop the precedence constraints ordering.py recorded for
    # `robot_id` across `indices` -- see ReleaseSchedule.release_override.
    release_override: Callable[[str, range], None]
    # (robot_id, start, goal) -> path points, or None if infeasible.
    plan_single: Callable[[str, Point, Point], list[Point] | None]
    # (starts, pinned_priority | None) -> True if the fleet was replanned
    # and normal dispatch resumed, False if the replan is infeasible.
    replan: Callable[[dict[str, Point], dict[str, float] | None], bool]
    fail: Callable[[str], None]
    log: Callable[[str], None]
    warn: Callable[[str], None]


class RecoveryManager:
    def __init__(
        self,
        *,
        priorities: dict[str, float],
        effective_radii: dict[str, float],
        hooks: RecoveryHooks,
        clearance_factor: float = 1.0,
        cfg: RecoveryConfig = RecoveryConfig(),
    ) -> None:
        self._priorities = dict(priorities)  # every robot in the fleet
        self._radii = dict(effective_radii)
        self._h = hooks
        self._clearance_factor = clearance_factor
        self._cfg = cfg

        self._phase = "idle"  # idle | yielding | replanning | failed
        self._group: frozenset[str] = frozenset()
        self._attempt = 0
        self._mode = ""  # "light" | "heavy", for logging/introspection only
        self._winner: str | None = None
        self._yielders: list[str] = []
        self._bottleneck: StructuralBottleneck | None = None
        self._pending: set[str] = set()
        self._yield_results: dict[str, bool] = {}
        self._yield_retry_count: dict[str, int] = {}
        self._on_yields_clear: Callable[[dict[str, bool]], None] | None = None

    def is_recovering(self) -> bool:
        return self._phase in ("yielding", "replanning")

    @property
    def failed(self) -> bool:
        return self._phase == "failed"

    def on_deadlock(self, group: Iterable[str]) -> None:
        """Entry point: coordinator_node calls this when it detects a mutual
        block. Re-reports while a recovery is already running are ignored;
        a fresh report for the *same* group after a completed recovery
        counts as the next attempt.
        """
        group = frozenset(group)
        if len(group) < 2 or self._phase == "failed" or self.is_recovering():
            return

        if group == self._group:
            self._attempt += 1
        else:
            self._group, self._attempt = group, 1

        if self._attempt > self._cfg.max_recovery_attempts:
            self._phase = "failed"
            self._h.fail(
                f"deadlock among {sorted(group)} survived "
                f"{self._cfg.max_recovery_attempts} recovery attempts"
            )
            return

        self._begin(group)

    # ── internals ──────────────────────────────────────────────────────────

    def _begin(self, group: frozenset[str]) -> None:
        bottleneck = None
        if len(group) == 2 and self._attempt == 1:
            a, b = sorted(group)
            bottleneck = self._h.structural_bottleneck(a, b)

        if bottleneck is not None:
            self._begin_light(bottleneck)
        else:
            self._begin_heavy(group)

    def _begin_light(self, bottleneck: StructuralBottleneck) -> None:
        self._mode = "light"
        self._bottleneck = bottleneck
        self._winner, self._yielders = bottleneck.first, [bottleneck.second]
        self._h.log(
            f"recovery attempt {self._attempt}/{self._cfg.max_recovery_attempts}: "
            f"structural bottleneck {bottleneck.first}/{bottleneck.second} -- "
            f"clearing just the shared zone, no replan"
        )
        self._h.hold(bottleneck.first)
        self._h.hold(bottleneck.second)
        self._phase = "yielding"
        # first's *entire* remaining route, not just zone_points -- see
        # class docstring for why the narrower set let a yield point back
        # onto first's path further along, past the zone itself. If that
        # can't be satisfied at all, _finish_light_wide retries against
        # just the zone -- see there for why a hard requirement here would
        # be worse than the risk it's guarding against.
        self._dispatch_yields(
            {bottleneck.second: self._h.remaining_of(bottleneck.first)},
            self._finish_light_wide,
        )

    def _finish_light_wide(self, results: dict[str, bool]) -> None:
        bottleneck = self._bottleneck
        assert bottleneck is not None
        if results.get(bottleneck.second, False):
            self._finish_light(bottleneck)
            return

        # No point anywhere is clear of first's *entire* remaining route --
        # on a tight map that's not necessarily a dead end, it can just mean
        # first's route structurally blankets most of the shared area (the
        # very definition of "structural"), so a search that wide can come
        # up empty even where the immediate conflict has a perfectly good
        # answer.
        self._h.warn(
            f"recovery: no yield for {bottleneck.second} clear of "
            f"{bottleneck.first}'s whole remaining route -- retrying "
            "against just the shared zone (reduced confidence)"
        )
        self._dispatch_yields(
            {bottleneck.second: list(bottleneck.zone_points)},
            self._finish_light_narrow,
        )

    def _finish_light_narrow(self, results: dict[str, bool]) -> None:
        bottleneck = self._bottleneck
        assert bottleneck is not None
        if not results.get(bottleneck.second, False):
            self._h.warn(
                f"recovery: couldn't confirm {bottleneck.second} clear of the "
                f"shared zone -- falling back to a full replan"
            )
            self._begin_heavy(frozenset((bottleneck.first, bottleneck.second)))
            return
        self._finish_light(bottleneck)

    def _finish_light(self, bottleneck: StructuralBottleneck) -> None:
        self._h.release_override(bottleneck.first, bottleneck.first_range)
        self._h.resume(bottleneck.first)
        self._h.resume(bottleneck.second)
        self._h.log(
            f"recovery: {bottleneck.first} resumes its own path through the "
            f"cleared zone; {bottleneck.second} resumes its own once "
            f"{bottleneck.first} actually clears it -- no replan needed"
        )
        self._phase = "idle"

    def _begin_heavy(self, group: frozenset[str]) -> None:
        self._mode = "heavy"
        remaining = {rid: len(self._h.remaining_of(rid)) for rid in group}
        self._winner, self._yielders = pick_winner(group, remaining, self._priorities)
        self._h.log(
            f"recovery attempt {self._attempt}/{self._cfg.max_recovery_attempts} for "
            f"{sorted(group)}: winner {self._winner} "
            f"({remaining[self._winner]} pts to goal), yielding "
            f"{self._yielders}, full replan"
        )
        for rid in group:
            self._h.hold(rid)
        self._phase = "yielding"
        winner_remaining = self._h.remaining_of(self._winner)
        self._dispatch_yields(
            {rid: winner_remaining for rid in self._yielders},
            lambda _results: self._do_replan(),
        )

    def _dispatch_yields(
        self,
        jobs: dict[str, list[Point]],
        on_all_clear: Callable[[dict[str, bool]], None],
    ) -> None:
        """Compute and dispatch a yield move for every (robot_id ->
        obstacles) in `jobs`; call `on_all_clear({robot_id: confirmed_clear})`
        once every job has either arrived, been given up on after
        cfg.max_yield_retries, or never had a yield pose to begin with.

        Every job is computed and _pending populated *before* dispatching
        any of them -- drive_oneoff can fire its callback synchronously
        (e.g. the action server isn't up), and the callback must not see a
        half-built _pending and finish early.
        """
        self._on_yields_clear = on_all_clear
        self._yield_results = {}
        self._yield_retry_count = {}

        plans: list[tuple[str, list[Point], list[Pose], str]] = []
        for rid, obstacles in jobs.items():
            poses, reason = self._yield_plan(rid, obstacles)
            if poses is None:
                self._h.warn(
                    f"recovery: no clear yield pose for {rid} -- leaving it put"
                )
                self._yield_results[rid] = False
                continue
            plans.append((rid, obstacles, poses, reason))

        self._pending = {rid for rid, _, _, _ in plans}
        if not self._pending:
            self._on_yields_clear(dict(self._yield_results))
            return
        for rid, obstacles, poses, reason in plans:
            self._h.drive_oneoff(
                rid, poses, self._make_yield_cb(rid, obstacles), reason
            )

    def _make_yield_cb(
        self, rid: str, obstacles: list[Point]
    ) -> Callable[[bool], None]:
        def cb(success: bool) -> None:
            if not success:
                retries = self._yield_retry_count.get(rid, 0)
                if retries < self._cfg.max_yield_retries:
                    self._yield_retry_count[rid] = retries + 1
                    self._h.warn(
                        f"recovery: {rid}'s yield move didn't complete -- "
                        f"retrying ({retries + 1}/{self._cfg.max_yield_retries})"
                    )
                    poses, reason = self._yield_plan(rid, obstacles)
                    if poses is not None:
                        self._h.drive_oneoff(
                            rid, poses, self._make_yield_cb(rid, obstacles), reason
                        )
                        return
                self._h.warn(
                    f"recovery: {rid} could not be confirmed clear after retries"
                )
            self._yield_results[rid] = success
            self._pending.discard(rid)
            if self._phase == "yielding" and not self._pending:
                on_all_clear = self._on_yields_clear
                assert on_all_clear is not None
                on_all_clear(dict(self._yield_results))

        return cb

    def _yield_plan(
        self, rid: str, obstacles: list[Point]
    ) -> tuple[list[Pose] | None, str]:
        clearance = self._clearance_factor * (
            self._radii[rid] + self._radii[self._winner]
        )
        here = self._h.position_of(rid)

        # 1 & 2. The two FREE moves (no Spooky call) -- retreat along the
        # trail it just drove, or advance along its own remaining route --
        # computed *both*, then the shorter real detour wins. A fixed order
        # (retreat, then sidestep, then advance last as a "resort") can
        # miss a cheap, real-progress advance that clears the conflict
        # outright while retreat only backs onto a point that still needs
        # a further sidestep.
        candidates: list[tuple[float, list[Pose], str, Point]] = []

        trail = self._h.trail_of(rid)
        hit = trail_retreat_point(trail, obstacles, clearance)
        if hit is not None:
            idx, pt = hit
            candidates.append(
                (distance(here, pt), retreat_poses(trail, idx), "recovery-retreat", pt)
            )

        remaining = self._h.remaining_of(rid)
        adv = forward_advance_point(remaining, obstacles, clearance)
        if adv is not None:
            idx, pt = adv
            candidates.append(
                (
                    distance(here, pt),
                    forward_poses(remaining[: idx + 1]),
                    "recovery-advance",
                    pt,
                )
            )

        if candidates:
            candidates.sort(key=lambda c: c[0])
            dist, poses, reason, pt = candidates[0]
            verb = (
                "backs up along its trail"
                if reason == "recovery-retreat"
                else ("advances along its own route")
            )
            self._h.log(
                f"recovery: {rid} {verb} to ({pt[0]:.2f}, {pt[1]:.2f}) "
                f"({dist:.2f} m)"
            )
            return poses, reason

        # 3. Neither free move works -- ask Spooky for a short hop to a
        #    clear cell to the side. Validated over the WHOLE returned
        #    route (path_never_worsens), not just the endpoint: a route
        #    that dips back toward the winner partway to an otherwise-clear
        #    goal hits the same "collision ahead" mid-transit the
        #    controller would on any other path through the contested
        #    space.
        step = self._radii[rid] + self._radii[self._winner]
        route_dir = _local_route_dir(obstacles)
        probes = sideways_probe_goals(here, route_dir, step)[
            : self._cfg.max_sidestep_probes
        ]
        for goal in probes:
            path = self._h.plan_single(rid, here, goal)
            if not path or not path_never_worsens(path, obstacles, clearance):
                continue
            end = path[-1]
            self._h.log(f"recovery: {rid} steps aside to ({end[0]:.2f}, {end[1]:.2f})")
            return forward_poses(path), "recovery-sidestep"

        return None, ""

    def _do_replan(self) -> None:
        if self._phase != "yielding":
            return  # already past this (a synchronous arrival got here first)
        self._phase = "replanning"
        starts = {rid: self._h.position_of(rid) for rid in self._priorities}
        pinned = (
            {self._winner: self._cfg.pinned_winner_priority}
            if self._attempt >= 2 and self._winner is not None
            else None
        )
        if pinned:
            self._h.log(
                f"recovery: replanning with {self._winner}'s priority pinned "
                "high -- forces the solver to route the others around it"
            )
        self._finish_replan(self._h.replan(starts, pinned))

    def _finish_replan(self, ok: bool) -> None:
        if ok:
            self._h.log("recovery: replan accepted -- normal dispatch resumes")
            self._phase = "idle"
        else:
            self._phase = "failed"
            self._h.fail("replan from current poses is infeasible")
