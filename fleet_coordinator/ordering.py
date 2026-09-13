"""Release-gating: turn Spooky's symbolic step ordering into a dispatch
schedule, using conflict-*zone* (reservation) gating.

Zone gating approach: find the whole contiguous band of index pairs where
two robots interact, pick ONE order for that pair (whichever robot enters
the band at the earlier symbolic step goes first; ties broken by robot
id), and hold the second robot at the mouth of the band until the first
has driven a full clearance distance *past the far end* of it -- not just
past its last conflicting cell. The second robot then gets the entire rest
of its path released at once. One robot traverses the shared region while
the other waits, genuinely clear of it. Less concurrency than per-cell
gating, but the hand-off pose is actually clear, which per-cell
gating never guaranteed.

Why this needs no MPC change: "waiting" isn't a mode the local planner has
to implement, it's what naturally happens when the path it's tracking
stops getting longer. released_path() returns a robot's currently-released
path *prefix*, growing over time as gates open; dispatch.py republishes it
as a continuous nav_msgs/Path each time it grows. A robot's MPC just keeps
tracking whatever the latest Path is and comes to rest at its end --
indistinguishable, from the MPC's point of view, from having reached a
real final goal. No
stop/resume signal, no special "hold" state to add to the controller.

path index == symbolic step: Spooky returns exactly one path point per
symbolic step t (reported steps == len(path)), so a point's list index IS
its t. The per-pair order is decided on those indices, and two points only
count as a conflict when their steps are within OrderingConfig's
conflict_step_window of each other as well as within combined effective
radius -- otherwise the symbolic plan already separates them in time and
only a gross timing failure (which the dispatch retry cap surfaces) could
bring them together. The safety guarantee still only holds *if* the
chosen relative order is preserved in real execution -- which is the
whole reason this gate exists, since per-robot MPC timing is never
pinned to the symbolic step pacing.
"""

from __future__ import annotations

import itertools
import logging
from collections import defaultdict
from dataclasses import dataclass, field

from .geometry import distance

logger = logging.getLogger(__name__)

# Two index pairs belong to the same conflict zone if their indices are
# within this many steps of each other on *both* robots -- i.e. the zone is
# a contiguous run, tolerating a one-step hole (a path that briefly steps
# just outside combined effective radius and back). Anything sparser is a
# genuinely separate encounter and gets its own zone.
_ZONE_MERGE_GAP = 1

# OrderingConfig.conflict_step_window: a spatially-close pair of path
# points (robot A at step i, robot B at step j) is only a real
# co-occupancy risk if the two robots are scheduled to be there at
# roughly the same time -- |i - j| within this many symbolic steps.
# Wider than that, the symbolic plan itself already keeps them apart (one
# is long gone before the other arrives) and the only remaining hazard is
# gross execution-timing failure, which the dispatch retry cap +
# on_stuck path surfaces instead. Without this bound, two robots on
# near-mirror routes (A's start cell ~ B's goal cell and vice versa) look
# "in conflict" end to end and collapse into one mission-long zone that
# serialises them completely. Assumes all robots start together
# (coordinator_node enforces this -- see its convergence gate).

# A pair is a *structural* bottleneck when clearing the zone alone isn't
# enough: the first robot has to physically travel OrderingConfig's
# structural_overhang_factor combined effective radii *past the zone's
# own far end* (real path distance, not a step count) before the second
# robot is released.
#
# Measuring the actual distance the first robot must drive fixes both:
# it means the same physical thing regardless of grid pitch, and a short
# remaining trip only fails to trigger this when there genuinely isn't
# much distance left to be at risk over.


@dataclass(frozen=True)
class OrderingConfig:
    """Release-gating tuning for ReleaseSchedule.from_fleet_plan -- see its
    own docstring for what each field means and how to tune it.
    """

    clearance_factor: float = 1.0
    conflict_step_window: int = 5
    # Tunable like clearance_factor. Widen it if real runs show false
    # positives on brief crossings, narrow it if a genuine structural
    # pair still releases the first robot in full.
    structural_overhang_factor: float = 1.2


def _path_length(path: list[tuple[float, float]], lo: int, hi: int) -> float:
    """Cumulative real distance actually driven along path[lo..hi]
    (inclusive), i.e. the sum of consecutive segment lengths -- not the
    straight-line distance between the endpoints, since a robot drives the
    path, not a shortcut across it."""
    return sum(distance(path[k], path[k + 1]) for k in range(lo, hi))


@dataclass(frozen=True)
class StructuralBottleneck:
    """One structural pair, as flagged by _build_constraints: `first`'s own
    route keeps `second` gated for essentially its whole remaining trip, so
    ordering.py gates BOTH at the shared stretch's mouth instead of
    releasing `first` unconditionally (see OrderingConfig.structural_overhang_factor).

    Consumed by recovery.py's *light* recovery path: once `second` is
    independently confirmed clear of `first`'s entire remaining route (not
    just `zone_points` -- a structural pair is defined by `first` staying
    close to `second` well past the zone too, so the wider set is what
    recovery actually checks the yield pose against; `zone_points` is kept
    here as the zone's own extent, e.g. for logging/diagnostics), a
    one-off sidestep/retreat, not a full replan --
    `release_override(first, first_range)` drops the very constraints
    recorded here, letting `first` resume its own already-planned path
    through the zone with no new plan. `second`'s own constraint (waiting
    on `first`'s real progress) is untouched -- it opens on its own, the
    normal way, once `first` actually gets there.
    """

    first: str
    second: str
    first_range: range  # indices on first's path to release_override
    zone_points: tuple[tuple[float, float], ...]  # first's points across the zone


@dataclass
class ReleaseSchedule:
    """Tracks, per robot, how much of its Spooky path is currently released
    for dispatch, given every other robot's reported progress and the
    per-pair zone-precedence constraints derived from their combined paths.

    Construct via `ReleaseSchedule.from_fleet_plan(...)`, not directly
    """

    _paths: dict[str, list[tuple[float, float]]]
    _constraints: dict[tuple[str, int], list[tuple[str, int]]]
    _structural: dict[frozenset[str], StructuralBottleneck] = field(
        default_factory=dict
    )
    _released_index: dict[str, int] = field(default_factory=dict)
    _cleared_index: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for robot_id in self._paths:
            self._released_index.setdefault(robot_id, -1)
            self._cleared_index.setdefault(robot_id, -1)

    @classmethod
    def from_fleet_plan(
        cls,
        robot_plans: dict,
        effective_radii: dict[str, float],
        cfg: OrderingConfig = OrderingConfig(),
    ) -> ReleaseSchedule:
        """Build a ReleaseSchedule from spooky_client.plan_fleet's result.

        Args:
            robot_plans: FleetPlan.robot_plans (robot_id -> RobotPlan).
            effective_radii: robot_id -> Robot.effective_radius
                (robot_radius + inflation, from the same Fleet the plan was
                requested for).
            cfg: release-gating tuning -- see OrderingConfig.
                clearance_factor: how far past a conflict zone the first
                    robot must get before the second is released into it,
                    as a multiple of the pair's combined effective radius.
                    1.0 (the default) means "one full combined effective
                    radius clear of every point in the shared band." Raise
                    it if the plan's hold poses still end up too close for
                    the controllers to clear each other; it scales with
                    the effective radius, so bumping robot_radius or
                    inflation in the mission alone also widens this.
                conflict_step_window: max |step_i - step_j| for a
                    spatially close pair of path points to count as a real
                    conflict. Raise it if execution timing drifts far
                    enough that a genuine encounter is being missed; lower
                    it if unrelated stretches of two routes get
                    serialised.
                structural_overhang_factor: how many combined effective
                    radii of real distance past the zone's far end the
                    first robot has to travel before the pair is treated
                    as a structural bottleneck (both gated at the zone
                    mouth) rather than releasing the first robot in full.

        All robots must share the same coordinate_format: distances
        between two robots' paths are meaningless if their positions are in
        different unit spaces.
        """
        formats = {rid: rp.coordinate_format for rid, rp in robot_plans.items()}
        if len(set(formats.values())) > 1:
            raise ValueError(
                f"ReleaseSchedule: robots use mixed coordinate_formats "
                f"({formats}) -- distances between them aren't comparable"
            )

        paths = {rid: rp.path for rid, rp in robot_plans.items()}
        constraints, structural = _build_constraints(
            paths,
            effective_radii,
            cfg.clearance_factor,
            cfg.conflict_step_window,
            cfg.structural_overhang_factor,
        )

        cycle = _precedence_cycle(constraints)
        if cycle and frozenset(cycle) in structural:
            logger.warning(
                "ReleaseSchedule: structural bottleneck between %s -- the "
                "first robot's own route stays close to the second's for "
                "most of its remaining trip, so both are gated at the shared "
                "stretch's mouth instead of releasing the first robot in "
                "full. They advance there and stop; the coordinator's stall "
                "detector hands that to deadlock recovery, which can clear "
                "just the zone (release_override) without a full replan.",
                " and ".join(sorted(frozenset(cycle))),
            )
        elif cycle:
            logger.error(
                "ReleaseSchedule: cyclic cross-robot precedence %s -- these "
                "robots will deadlock, each gated behind the next. Release-"
                "gating cannot resolve this; it needs a replan with consistent "
                "priorities.",
                " -> ".join(cycle),
            )

        return cls(_paths=paths, _constraints=constraints, _structural=structural)

    def released_path(self, robot_id: str) -> list[tuple[float, float]]:
        """Return the path prefix `robot_id` is currently released to
        track, from its start up to (and including) the furthest index not
        blocked by an unsatisfied precedence constraint. Grows over time as
        report_progress() unblocks more of it -- callers (dispatch.py)
        should republish this as an extended nav_msgs/Path each time it's
        longer than what they last sent, not diff it point-by-point.
        """
        path = self._paths[robot_id]
        idx = self._released_index[robot_id]
        while idx + 1 < len(path) and self._is_released(robot_id, idx + 1):
            idx += 1
        self._released_index[robot_id] = idx
        return path[: idx + 1]

    def report_progress(self, robot_id: str, reached_index: int) -> None:
        """Record that `robot_id` has actually reached (index) `reached_index`
        of its own path -- may unblock other robots waiting on it. Mapping a
        live pose to a path index (nearest point, arc-length progress, ...)
        is dispatch.py's job, not this module's -- this only tracks indices.
        """
        self._cleared_index[robot_id] = max(
            self._cleared_index[robot_id], reached_index
        )

    def pending_blockers(self, robot_id: str) -> list[tuple[str, int, int]]:
        """The still-unsatisfied precedence constraints holding `robot_id`
        at its current release boundary: one
        (other_robot, required_index, other_robot's current cleared_index)
        tuple per blocker. Empty when `robot_id` isn't gated (its whole
        path is released, or it's simply at the end).

        Diagnostic only -- dispatch/coordinator use it to report *why* a
        robot is waiting when it's been stuck a while. Reads the release
        boundary left by the last released_path() call (the dispatch tick
        refreshes that every period), so it can lag reality by one tick;
        fine for a human-facing log line.
        """
        path = self._paths[robot_id]
        next_index = self._released_index[robot_id] + 1
        if next_index >= len(path):
            return []
        return [
            (other, other_index, self._cleared_index[other])
            for other, other_index in self._constraints.get((robot_id, next_index), ())
            if self._cleared_index[other] < other_index
        ]

    def structural_bottleneck(
        self, robot_a: str, robot_b: str
    ) -> StructuralBottleneck | None:
        """The StructuralBottleneck between these two robots, if any -- used
        by recovery.py to attempt the light fix (clear just the zone) before
        falling back to a full replan. None for any other pair, including
        ones with an ordinary (non-structural) precedence constraint.
        """
        return self._structural.get(frozenset((robot_a, robot_b)))

    def release_override(self, robot_id: str, indices: range) -> None:
        """Recovery-only escape hatch: drop the precedence constraints on
        `robot_id` for `indices` outright, without touching _cleared_index
        (which other robots' constraints may still depend on for real).

        Use only once something outside this module has independently
        confirmed the party those constraints depended on is actually
        clear -- e.g. recovery.py physically moving a structural
        bottleneck's `second` robot out of `first`'s zone, then overriding
        exactly `first_range` from that same StructuralBottleneck. Blindly
        overriding a constraint that hasn't really been resolved defeats
        the whole point of this module.
        """
        for index in indices:
            self._constraints.pop((robot_id, index), None)

    def _is_released(self, robot_id: str, index: int) -> bool:
        for other_robot, other_index in self._constraints.get((robot_id, index), ()):
            if self._cleared_index[other_robot] < other_index:
                return False
        return True


@dataclass
class _Zone:
    """A contiguous band of index pairs where two robots' paths stay within
    combined effective radius: [a_lo, a_hi] on robot A, [b_lo, b_hi] on B.
    """

    a_lo: int
    a_hi: int
    b_lo: int
    b_hi: int


def _cluster_zones(close_pairs: list[tuple[int, int]]) -> list[_Zone]:
    """Group (i, j) index pairs into conflict zones -- two pairs share a
    zone when their indices are within _ZONE_MERGE_GAP on both axes
    (union-find over that relation). O(n^2) in the number of close pairs,
    which is fine for realistic path lengths (same reasoning as the
    conflict scan that produced them).
    """
    parent = list(range(len(close_pairs)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for p in range(len(close_pairs)):
        ip, jp = close_pairs[p]
        for q in range(p + 1, len(close_pairs)):
            iq, jq = close_pairs[q]
            if abs(ip - iq) <= _ZONE_MERGE_GAP and abs(jp - jq) <= _ZONE_MERGE_GAP:
                parent[find(p)] = find(q)

    groups: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for p, pair in enumerate(close_pairs):
        groups[find(p)].append(pair)

    zones: list[_Zone] = []
    for members in groups.values():
        rows = [i for i, _ in members]
        cols = [j for _, j in members]
        zones.append(_Zone(min(rows), max(rows), min(cols), max(cols)))
    return zones


def _pose_clear(
    pose: tuple[float, float],
    points: list[tuple[float, float]],
    clear_distance: float,
) -> bool:
    """True when `pose` is at least `clear_distance` from every point in
    `points`."""
    return all(distance(pose, point) >= clear_distance for point in points)


def _clear_target(
    first_path: list[tuple[float, float]],
    first_hi: int,
    second_segment: list[tuple[float, float]],
    clear_distance: float,
) -> int:
    """Smallest index t >= first_hi at which first_path[t] is at least
    `clear_distance` from *every* point in the second robot's zone segment
    -- i.e. the first robot has physically pulled clear of the whole shared
    stretch, not just past its own last conflicting cell.

    Falls back to the last index (the first robot must finish its entire
    path before the second is let in) when its path never gets that far
    clear -- e.g. it curves back alongside the shared stretch.
    """
    for t in range(first_hi, len(first_path)):
        if _pose_clear(first_path[t], second_segment, clear_distance):
            return t
    return len(first_path) - 1


def _gate_index(
    second_path: list[tuple[float, float]],
    zone_entry: int,
    first_transit: list[tuple[float, float]],
    clear_distance: float,
) -> int:
    """The index to actually hold the second robot before -- normally the
    zone entry, but pulled earlier while the hold pose it would rest at
    (gate - 1) is itself within `clear_distance` of anywhere the first
    robot drives on its way through and clear of the zone (`first_transit`).
    """
    gate = zone_entry
    while gate > 0 and not _pose_clear(
        second_path[gate - 1], first_transit, clear_distance
    ):
        gate -= 1
    return gate


def _build_constraints(
    paths: dict[str, list[tuple[float, float]]],
    effective_radii: dict[str, float],
    clearance_factor: float,
    conflict_step_window: int,
    structural_overhang_factor: float,
) -> tuple[
    dict[tuple[str, int], list[tuple[str, int]]],
    dict[frozenset[str], StructuralBottleneck],
]:
    """For every pair of robots: find the conflict zones between their
    paths, choose one traversal order for the pair, and emit a precedence
    constraint for every index of the *second* robot from its hold gate
    onward -- (second, k) -> (first, clear_target) -- so released_path()
    holds the second robot before the zone until the first has cleared the
    whole zone by `clearance_factor * combined_radius`. The gate is the
    zone entry, pulled earlier if resting there would still leave the
    second robot within clear_distance of the first robot's track
    (_gate_index).

    Structural exception: two signals mean gating the second robot alone
    just parks it while the first races off and wedges later with no
    earlier warning -- either is enough:
      - _gate_index pulled the second robot's gate all the way back to 0:
        its own start isn't even clear of the first robot's transit.
      - the first robot has to travel structural_overhang_factor combined
        effective radii or more of real distance past the zone's own far
        end before the second robot's wait target is reached: even though
        the second robot found a fine hold pose, the first robot's route
        keeps running close to it well beyond the crossing itself.
    Either way, BOTH robots are gated at their zone entries instead,
    against each other. They advance to opposite mouths of the shared
    stretch, stop clean, and the coordinator's stall detector hands that
    symmetric mid-route state to deadlock recovery -- which can clear just
    the zone (ReleaseSchedule.release_override) rather than a full replan.
    Such pairs are returned as StructuralBottleneck records, keyed by
    frozenset(pair), so from_fleet_plan can phrase the (expected)
    precedence cycle as a bottleneck, not a bug, and recovery.py can find
    them.

    A pair of path points counts as conflicting only when it's both within
    combined effective radius *and* within conflict_step_window symbolic
    steps (see OrderingConfig).

    The order is fixed per pair (not per zone): whichever robot enters the
    earliest-starting zone at the lower symbolic step goes first,
    everywhere. A per-zone order could otherwise leapfrog into a precedence
    cycle between just two robots.
    """
    constraints: dict[tuple[str, int], list[tuple[str, int]]] = defaultdict(list)
    structural: dict[frozenset[str], StructuralBottleneck] = {}

    for robot_a, robot_b in itertools.combinations(sorted(paths), 2):
        path_a, path_b = paths[robot_a], paths[robot_b]
        combined_radius = effective_radii[robot_a] + effective_radii[robot_b]
        clear_distance = clearance_factor * combined_radius

        close_pairs = [
            (i, j)
            for i, point_a in enumerate(path_a)
            for j, point_b in enumerate(path_b)
            if abs(i - j) <= conflict_step_window
            and distance(point_a, point_b) <= combined_radius
        ]
        if not close_pairs:
            continue

        zones = _cluster_zones(close_pairs)

        # One order for the whole pair, taken from the zone that starts
        # earliest (lowest entry step on either robot).
        earliest = min(zones, key=lambda z: (min(z.a_lo, z.b_lo), z.a_lo, z.b_lo))
        a_goes_first = earliest.a_lo <= earliest.b_lo
        if earliest.a_lo == earliest.b_lo:
            logger.warning(
                "ReleaseSchedule: robots %r and %r enter a shared zone at the "
                "same symbolic step %d (combined effective radius %.3f) -- "
                "Spooky's own plan may have a same-step conflict; tie-breaking "
                "by robot id (%r first)",
                robot_a,
                robot_b,
                earliest.a_lo,
                combined_radius,
                robot_a,
            )

        for zone in zones:
            if a_goes_first:
                first, first_path = robot_a, path_a
                first_lo, first_hi = zone.a_lo, zone.a_hi
                second, second_path = robot_b, path_b
                second_lo, second_hi = zone.b_lo, zone.b_hi
            else:
                first, first_path = robot_b, path_b
                first_lo, first_hi = zone.b_lo, zone.b_hi
                second, second_path = robot_a, path_a
                second_lo, second_hi = zone.a_lo, zone.a_hi

            second_segment = second_path[second_lo : second_hi + 1]
            target = _clear_target(first_path, first_hi, second_segment, clear_distance)
            gate = _gate_index(
                second_path,
                second_lo,
                first_path[first_lo : target + 1],
                clear_distance,
            )

            no_hold_pose = gate == 0 and second_lo > 0
            overhang_distance = _path_length(first_path, first_hi, target)
            long_overhang = overhang_distance >= structural_overhang_factor * (
                combined_radius
            )
            if no_hold_pose or long_overhang:
                # Structural: gate the first robot too, at its own zone
                # entry, so both advance to the shared stretch's mouth and
                # stop (see docstring).
                first_range = range(first_lo, first_hi + 1)
                first_segment = first_path[first_lo : first_hi + 1]
                structural[frozenset((first, second))] = StructuralBottleneck(
                    first=first,
                    second=second,
                    first_range=first_range,
                    zone_points=tuple(first_segment),
                )
                first_target = _clear_target(
                    second_path, second_hi, first_segment, clear_distance
                )
                for k in first_range:
                    constraints[(first, k)].append((second, first_target))
                gate = second_lo

            for k in range(gate, second_hi + 1):
                constraints[(second, k)].append((first, target))

    return dict(constraints), structural


def _precedence_cycle(
    constraints: dict[tuple[str, int], list[tuple[str, int]]],
) -> list[str] | None:
    """Return a robot-id cycle in the "waits for" graph (an edge second ->
    first for every (second, _) -> (first, _) constraint), or None. A cycle
    means those robots mutually block and will deadlock.
    """
    graph: dict[str, set[str]] = defaultdict(set)
    for (robot, _), deps in constraints.items():
        for other, _ in deps:
            if other != robot:
                graph[robot].add(other)

    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = defaultdict(int)
    path: list[str] = []

    def visit(node: str) -> list[str] | None:
        color[node] = GRAY
        path.append(node)
        for nxt in graph[node]:
            if color[nxt] == GRAY:
                return path[path.index(nxt) :] + [nxt]
            if color[nxt] == WHITE:
                found = visit(nxt)
                if found:
                    return found
        path.pop()
        color[node] = BLACK
        return None

    for node in list(graph):
        if color[node] == WHITE:
            found = visit(node)
            if found:
                return found
    return None
