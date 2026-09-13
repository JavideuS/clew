"""Unit tests for recovery.py -- the pure decision logic and the FSM
driven through fake hooks (no ROS)."""

from __future__ import annotations

import math

import pytest

from fleet_coordinator.ordering import StructuralBottleneck
from fleet_coordinator.recovery import (
    RecoveryHooks,
    RecoveryManager,
    forward_advance_point,
    forward_poses,
    path_never_worsens,
    pick_winner,
    retreat_poses,
    sideways_probe_goals,
    trail_retreat_point,
)

# ─── pure logic ────────────────────────────────────────────────────────────


def test_pick_winner_prefers_fewest_remaining():
    winner, yielders = pick_winner(
        ["a", "b", "c"],
        remaining_counts={"a": 10, "b": 2, "c": 7},
        priorities={"a": 1.0, "b": 1.0, "c": 1.0},
    )
    assert winner == "b"
    assert set(yielders) == {"a", "c"}


def test_pick_winner_priority_then_id_tiebreak():
    # equal remaining -> higher priority wins
    winner, _ = pick_winner(
        ["a", "b"],
        remaining_counts={"a": 5, "b": 5},
        priorities={"a": 1.0, "b": 3.0},
    )
    assert winner == "b"
    # equal remaining and priority -> lower id wins
    winner, _ = pick_winner(
        ["b", "a"],
        remaining_counts={"a": 5, "b": 5},
        priorities={"a": 1.0, "b": 1.0},
    )
    assert winner == "a"


def test_trail_retreat_point_finds_nearest_backward_clear():
    trail = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]  # drove 0 -> 3
    # winner's route sits just past the end of the trail
    hit = trail_retreat_point(trail, obstacles=[(3.5, 0.0)], clearance=1.5)
    assert hit == (2, (2.0, 0.0))  # (3,0) is the end (skipped); (2,0) clears


def test_trail_retreat_point_none_when_whole_trail_conflicts():
    trail = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]
    assert trail_retreat_point(trail, obstacles=[(1.5, 0.0)], clearance=5.0) is None


def test_trail_retreat_point_none_when_adjacent_point_blocks_the_way_back():
    # (1,0), right before the current position, doesn't clear -- so nothing
    # farther back does either, even though (0,0) individually would: the
    # robot has to drive back THROUGH (1,0) to reach it.
    trail = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]
    assert trail_retreat_point(trail, obstacles=[(1.2, 0.0)], clearance=1.0) is None


def test_trail_retreat_point_none_for_single_point_trail():
    # robot never moved -- nowhere to retreat to
    assert trail_retreat_point([(0.0, 0.0)], obstacles=[(9.0, 9.0)], clearance=1.0) is None


def test_forward_advance_point_walks_to_furthest_prefix_clear():
    remaining = [(5.0, 5.0), (5.0, 3.5), (5.0, 2.0), (5.0, 0.5)]  # straight down, away
    hit = forward_advance_point(remaining, obstacles=[(3.0, 3.0)], clearance=1.5)
    assert hit == (3, (5.0, 0.5))  # every step clears -> go as far as the route does


def test_forward_advance_point_stops_before_route_re_enters_zone():
    remaining = [(5.0, 5.0), (5.0, 3.5), (4.0, 3.0), (3.2, 3.0)]  # curls back toward (3,3)
    hit = forward_advance_point(remaining, obstacles=[(3.0, 3.0)], clearance=1.5)
    assert hit == (1, (5.0, 3.5))  # (4,3) is 1.0 from (3,3) -> stop at (5,3.5)


def test_forward_advance_point_none_when_first_step_not_clear():
    remaining = [(5.0, 5.0), (4.6, 5.0), (4.0, 5.0)]  # first step heads at the winner
    assert forward_advance_point(remaining, obstacles=[(5.0, 5.0)], clearance=1.5) is None


def test_retreat_poses_reversed_with_forward_headings():
    trail = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]
    poses = retreat_poses(trail, retreat_idx=2)
    # current pose (3,0) first, retreat point (2,0) last
    assert [(x, y) for x, y, _ in poses] == [(3.0, 0.0), (2.0, 0.0)]
    # heading stays at the original +x forward tangent (not reversed to pi)
    assert all(theta == pytest.approx(0.0) for _, _, theta in poses)


def test_retreat_poses_empty_when_nothing_to_travel():
    assert retreat_poses([(0.0, 0.0), (1.0, 0.0)], retreat_idx=1) == []


def test_forward_poses_headings_are_tangents():
    poses = forward_poses([(0.0, 0.0), (1.0, 1.0), (2.0, 1.0)])
    assert poses[0][2] == pytest.approx(math.pi / 4)
    assert poses[1][2] == pytest.approx(0.0)
    assert poses[2][2] == pytest.approx(0.0)  # last inherits previous


def test_sideways_probe_goals_prefer_perpendicular_to_corridor():
    goals = sideways_probe_goals(
        center=(0.0, 0.0), route_dir=(5.0, 0.0), step=1.0, count=4,
        ring_factors=(1.0,),
    )
    assert len(goals) == 4
    # corridor runs along x -> the best escape is along y
    assert abs(goals[0][1]) > abs(goals[0][0])


def test_path_never_worsens_accepts_a_steady_departure():
    # starts 0.4 m from the obstacle (still inside the conflict -- that's
    # the point), but only ever gets farther, finishing well past clearance
    obstacle = [(0.0, 0.0)]
    path = [(0.4, 0.0), (0.6, 0.0), (1.0, 0.0), (1.6, 0.0), (2.4, 0.0)]
    assert path_never_worsens(path, obstacle, clearance=1.0)


def test_path_never_worsens_rejects_a_dip_back_even_with_a_clear_endpoint():
    # a route whose destination looks clear, but which drifts BACK toward
    # the obstacle partway there before recovering -- exactly the failure
    # mode a real run hit (a sidestep whose Spooky-planned route swung back
    # near the winner en route to an otherwise-clear goal).
    obstacle = [(0.0, 0.0)]
    path = [(0.6, 0.0), (0.5, 0.0), (0.3, 0.0), (1.5, 0.0)]
    assert not path_never_worsens(path, obstacle, clearance=1.0)


def test_path_never_worsens_rejects_never_reaching_clearance():
    # steadily improves, but never actually reaches the required clearance
    obstacle = [(0.0, 0.0)]
    path = [(0.4, 0.0), (0.5, 0.0), (0.6, 0.0)]
    assert not path_never_worsens(path, obstacle, clearance=1.0)


def test_path_never_worsens_false_for_single_point_path():
    assert not path_never_worsens([(2.0, 0.0)], [(0.0, 0.0)], clearance=1.0)


# ─── FSM through fake hooks ────────────────────────────────────────────────


class FakeHooks:
    def __init__(
        self,
        *,
        trails,
        remainings,
        positions,
        plan_single=None,
        replan_ok=True,
        bottlenecks=None,
    ):
        self.trails = trails
        self.remainings = remainings
        self.positions = positions
        self._plan_single = plan_single
        self._replan_ok = replan_ok
        self._bottlenecks = bottlenecks or {}  # frozenset({a,b}) -> StructuralBottleneck
        self.held: list[str] = []
        self.resumed: list[str] = []
        self.overrides: list[tuple[str, range]] = []
        self.driven: list[tuple[str, list, str]] = []
        self.replans: list[tuple[dict, dict | None]] = []
        self.fails: list[str] = []
        self.msgs: list[str] = []
        self._arrival: dict[str, callable] = {}

    def bundle(self) -> RecoveryHooks:
        return RecoveryHooks(
            position_of=lambda rid: self.positions[rid],
            trail_of=lambda rid: self.trails[rid],
            remaining_of=lambda rid: self.remainings[rid],
            drive_oneoff=self._drive,
            hold=lambda rid: self.held.append(rid),
            resume=lambda rid: self.resumed.append(rid),
            structural_bottleneck=lambda a, b: self._bottlenecks.get(frozenset((a, b))),
            release_override=lambda rid, indices: self.overrides.append((rid, indices)),
            plan_single=lambda rid, s, g: (
                self._plan_single(rid, s, g)
                if callable(self._plan_single)
                else self._plan_single
            ),
            replan=self._replan,
            fail=lambda m: self.fails.append(m),
            log=self.msgs.append,
            warn=self.msgs.append,
        )

    def _drive(self, rid, poses, on_done, reason):
        self.driven.append((rid, poses, reason))
        self._arrival[rid] = on_done

    def _replan(self, starts, pinned):
        self.replans.append((starts, pinned))
        return self._replan_ok

    def arrive(self, rid, success=True):
        self._arrival.pop(rid)(success)


def _manager(fh: FakeHooks, radii):
    return RecoveryManager(
        priorities={rid: 1.0 for rid in radii},
        effective_radii=radii,
        hooks=fh.bundle(),
    )


def test_retreat_then_unpinned_replan():
    fh = FakeHooks(
        trails={"sim1": [(0, 0), (1, 0)], "sim2": [(5, 5), (4, 5), (3, 5), (2, 5)]},
        remainings={
            "sim1": [(1, 0), (2, 0)],  # nearly done -> winner
            "sim2": [(2, 5), (1, 5), (0, 5), (0, 0), (-1, 0), (-2, 0)],
        },
        positions={"sim1": (1, 0), "sim2": (2, 5)},
    )
    rm = _manager(fh, {"sim1": 0.5, "sim2": 0.5})

    rm.on_deadlock(["sim1", "sim2"])
    assert set(fh.held) == {"sim1", "sim2"}
    assert rm.is_recovering()
    assert fh.driven and fh.driven[0][0] == "sim2"  # sim2 backs up
    assert fh.driven[0][2] == "recovery-retreat"
    assert not fh.replans  # not until the yield move finishes

    # re-report while already recovering -> ignored
    rm.on_deadlock(["sim1", "sim2"])
    assert len(fh.driven) == 1

    fh.arrive("sim2")
    assert fh.replans and fh.replans[0][1] is None  # attempt 1: no priority pin
    assert not rm.is_recovering()
    assert not fh.fails


def test_yielder_advances_when_own_route_leads_clear():
    # No trail to retreat along, no sidestep route, but the yielder's own
    # route heads straight away from the winner -> advance along it.
    fh = FakeHooks(
        trails={"w": [(0.0, 0.0)], "y": [(5.0, 5.0)]},
        remainings={
            "w": [(0.0, 0.0), (0.5, 0.0), (1.0, 0.0)],  # winner, stays lower-left
            "y": [(5.0, 5.0), (5.0, 4.0), (5.0, 3.0), (5.0, 2.0)],  # straight down
        },
        positions={"w": (0.0, 0.0), "y": (5.0, 5.0)},
        plan_single=None,  # no sidestep
    )
    rm = _manager(fh, {"w": 0.5, "y": 0.5})  # clearance 1.0

    rm.on_deadlock(["w", "y"])
    assert fh.driven and fh.driven[0][0] == "y"
    assert fh.driven[0][2] == "recovery-advance"
    last = fh.driven[0][1][-1]
    assert (last[0], last[1]) == (5.0, 2.0)
    fh.arrive("y")
    assert fh.replans and fh.replans[0][1] is None


def test_yielder_sidesteps_off_route_when_trail_wont_clear():
    # Near-mirror mid-corridor stop: no trail point clears, but Spooky can
    # route a short hop to the side that ends clear of the winner's route.
    def plan_single(rid, start, goal):
        return [start, ((start[0] + goal[0]) / 2, (start[1] + goal[1]) / 2), goal]

    fh = FakeHooks(
        trails={
            "w": [(1.0, 1.0), (2.0, 2.0)],
            "y": [(4.0, 4.0), (3.0, 3.0)],  # mirror of w, nothing clears
        },
        remainings={
            "w": [(2.0, 2.0), (3.0, 3.0), (4.0, 4.0), (5.0, 5.0)],
            "y": [(3.0, 3.0), (2.0, 2.0), (1.0, 1.0), (0.0, 0.0)],
        },
        positions={"w": (2.0, 2.0), "y": (3.0, 3.0)},
        plan_single=plan_single,
    )
    rm = _manager(fh, {"w": 0.5, "y": 0.5})  # clearance 1.0

    rm.on_deadlock(["w", "y"])
    assert fh.driven and fh.driven[0][0] == "y"
    assert fh.driven[0][2] == "recovery-sidestep"
    fh.arrive("y")
    assert fh.replans


def test_sidestep_rejects_a_dip_back_route_and_tries_the_next_probe():
    # A confirmed real-run failure: the first candidate hop's endpoint
    # looks clear, but Spooky's route to get there dips back toward the
    # winner partway through. That candidate must be rejected and the next
    # one tried, not accepted on its endpoint alone.
    calls: list[tuple] = []

    def plan_single(rid, s, g):
        calls.append(g)
        if len(calls) == 1:
            return [s, (1.0, 0.0), (4.0, 3.0)]  # dips toward (0,0)/(1,1) first
        return [s, (3.0, 1.5), (4.0, 3.0)]  # steady departure

    fh = FakeHooks(
        trails={"w": [(0.0, 0.0)], "y": [(2.5, 0.0)]},
        remainings={
            "w": [(0.0, 0.0), (1.0, 1.0)],
            "y": [(2.5, 0.0), (0.3, 0.0)],  # first advance step also fails
        },
        positions={"w": (0.0, 0.0), "y": (2.5, 0.0)},
        plan_single=plan_single,
    )
    rm = _manager(fh, {"w": 0.5, "y": 0.5})  # clearance 1.0

    rm.on_deadlock(["w", "y"])
    assert len(calls) == 2  # first probe rejected, second one used
    assert fh.driven and fh.driven[0][0] == "y"
    assert fh.driven[0][2] == "recovery-sidestep"
    assert fh.driven[0][1][-1][:2] == (4.0, 3.0)


def test_yield_prefers_the_shorter_of_retreat_or_advance():
    # y can both retreat (4 m back along its trail) and advance (1.5 m
    # forward, where its own route runs back into the winner's remaining
    # path and forward_advance_point stops) -- the shorter real detour
    # should win, not a fixed "retreat first" priority. This is the exact
    # case a real run got wrong: a comfortable forward move existed but a
    # farther retreat was tried (and failed) first.
    fh = FakeHooks(
        trails={"w": [(0.0, 0.0)], "y": [(10.0, 0.0), (9.0, 0.0), (5.0, 0.0)]},
        remainings={
            "w": [(0.0, 0.0), (1.0, 0.0), (5.0, 3.5)],
            "y": [(5.0, 0.0), (5.0, 1.5), (5.0, 3.5)],  # re-enters w's route at (5,3.5)
        },
        positions={"w": (0.0, 0.0), "y": (5.0, 0.0)},
    )
    rm = _manager(fh, {"w": 0.5, "y": 0.5})  # clearance 1.0

    rm.on_deadlock(["w", "y"])
    assert fh.driven and fh.driven[0][0] == "y"
    assert fh.driven[0][2] == "recovery-advance"
    assert fh.driven[0][1][-1][:2] == (5.0, 1.5)


def test_no_yield_pose_still_replans():
    # yielder's own route loops right back through the winner's path -- no
    # retreat, no advance, no sidestep.
    fh = FakeHooks(
        trails={"w": [(0, 0), (1, 0)], "y": [(2, 0), (2, 1), (2, 2)]},
        remainings={
            "w": [(1, 0), (2, 0)],
            "y": [(2, 2), (2, 1), (2, 0), (1, 0)],  # heads back onto w's route
        },
        positions={"w": (1, 0), "y": (2, 2)},
        plan_single=None,  # no sidestep route either
    )
    rm = _manager(fh, {"w": 1.5, "y": 1.5})  # clearance 3.0

    rm.on_deadlock(["w", "y"])
    assert not fh.driven  # nowhere to yield
    assert fh.replans  # replanned regardless
    assert any("no clear yield pose" in m for m in fh.msgs)


def test_escalates_to_pinned_then_fails():
    fh = FakeHooks(
        trails={"a": [(0, 0), (1, 0)], "b": [(9, 9), (8, 9)]},
        remainings={"a": [(1, 0)], "b": [(8, 9), (7, 9), (6, 9)]},
        positions={"a": (1, 0), "b": (8, 9)},
    )
    rm = _manager(fh, {"a": 0.5, "b": 0.5})

    rm.on_deadlock(["a", "b"])
    fh.arrive("b")
    assert fh.replans[-1][1] is None  # attempt 1

    rm.on_deadlock(["a", "b"])
    fh.arrive("b")
    assert fh.replans[-1][1] == {"a": pytest.approx(1_000_000.0)}  # attempt 2: winner pinned

    rm.on_deadlock(["a", "b"])  # attempt 3 -> give up
    assert fh.fails and rm.failed
    assert not rm.is_recovering()


def test_infeasible_replan_fails_immediately():
    fh = FakeHooks(
        trails={"a": [(0, 0), (1, 0)], "b": [(9, 9), (8, 9)]},
        remainings={"a": [(1, 0)], "b": [(8, 9), (7, 9)]},
        positions={"a": (1, 0), "b": (8, 9)},
        replan_ok=False,
    )
    rm = _manager(fh, {"a": 0.5, "b": 0.5})

    rm.on_deadlock(["a", "b"])
    fh.arrive("b")
    assert fh.fails and rm.failed


# ─── light path (structural bottleneck: clear the zone, no replan) ─────────


def test_light_path_overrides_and_resumes_without_a_replan():
    bottleneck = StructuralBottleneck(
        first="w",
        second="y",
        first_range=range(2, 5),
        zone_points=((2.0, 2.0), (3.0, 3.0), (4.0, 4.0)),
    )
    fh = FakeHooks(
        trails={"w": [(0, 0), (1, 1)], "y": [(9, 9), (8, 8)]},  # y can retreat
        remainings={
            "w": [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)],
            "y": [(8, 8), (7, 7)],
        },
        positions={"w": (1, 1), "y": (8, 8)},
        bottlenecks={frozenset(("w", "y")): bottleneck},
    )
    rm = _manager(fh, {"w": 0.5, "y": 0.5})

    rm.on_deadlock(["w", "y"])
    assert set(fh.held) == {"w", "y"}
    # only the second robot yields -- the first keeps its own already-planned
    # path, it just needs the zone physically cleared
    assert len(fh.driven) == 1
    assert fh.driven[0][0] == "y"
    assert not fh.replans  # no replan attempted yet

    fh.arrive("y")
    assert fh.overrides == [("w", bottleneck.first_range)]
    assert set(fh.resumed) == {"w", "y"}
    assert not fh.replans  # never needed one
    assert not rm.is_recovering()
    assert not fh.fails


def test_light_path_checks_yield_against_firsts_whole_remaining_route():
    # first's remaining route runs on past the zone_points segment to a
    # point that sits right next to second's only retreat/advance
    # candidate against the WIDE (whole-route) obstacle set -- exactly the
    # real-run failure this guards against (second parked 0.5 m off
    # first's path further down, well inside their combined radius, and
    # first deadlocked against it again a few seconds after resuming).
    # Checking only zone_points would call this candidate clear outright;
    # checking first's whole remaining route must not let it through on
    # the first pass. But the narrower zone_points set (checked second, as
    # a fallback) doesn't cover that far-out point either, so the same
    # retreat move IS confirmed clear against it -- the fallback finds the
    # same answer the old zone_points-only check would have, at reduced
    # confidence, rather than giving up outright.
    bottleneck = StructuralBottleneck(
        first="w",
        second="y",
        first_range=range(2, 4),
        zone_points=((2.0, 2.0), (3.0, 3.0)),  # narrow -- doesn't cover (9, 9.3)
    )
    fh = FakeHooks(
        trails={"w": [(0, 0), (1, 1)], "y": [(9.0, 9.0), (8.0, 8.0)]},
        remainings={
            # first's real remaining route: the zone, plus a further point
            # 0.3 m from y's only retreat candidate (9, 9) -- inside the
            # combined-radius clearance of 1.0. Blocks the wide check.
            "w": [(2.0, 2.0), (3.0, 3.0), (9.0, 9.3)],
            "y": [(9.0, 9.2), (9.4, 9.4), (9.8, 9.8), (10.0, 10.0)],
        },
        positions={"w": (1.0, 1.0), "y": (8.0, 8.0)},
        bottlenecks={frozenset(("w", "y")): bottleneck},
        plan_single=None,  # no sidestep needed -- retreat clears on the retry
    )
    rm = _manager(fh, {"w": 0.5, "y": 0.5})

    rm.on_deadlock(["w", "y"])
    # wide check found nothing driveable yet (synchronous failure) --
    # already retried and found a driveable retreat against the narrow set
    assert any("reduced confidence" in m for m in fh.msgs)
    assert len(fh.driven) == 1
    assert fh.driven[0][0] == "y"
    assert not fh.replans  # never needed the heavy path at all

    fh.arrive("y")
    assert fh.overrides == [("w", bottleneck.first_range)]
    assert set(fh.resumed) == {"w", "y"}
    assert not fh.replans
    assert not rm.is_recovering()
    assert not fh.fails


def test_light_path_falls_back_to_heavy_when_neither_obstacle_set_clears():
    # Same shape as above, but now even the narrow zone_points set blocks
    # every move too (nothing left anywhere) -- both passes must fail
    # before conceding to a full replan.
    bottleneck = StructuralBottleneck(
        first="w",
        second="y",
        first_range=range(2, 4),
        zone_points=((9.0, 9.3),),  # narrow set ALSO covers the blocking point
    )
    fh = FakeHooks(
        trails={"w": [(0, 0), (1, 1)], "y": [(9.0, 9.0), (8.0, 8.0)]},
        remainings={
            "w": [(2.0, 2.0), (3.0, 3.0), (9.0, 9.3)],
            "y": [(9.0, 9.2), (9.4, 9.4), (9.8, 9.8), (10.0, 10.0)],
        },
        positions={"w": (1.0, 1.0), "y": (8.0, 8.0)},
        bottlenecks={frozenset(("w", "y")): bottleneck},
        plan_single=None,
    )
    rm = _manager(fh, {"w": 0.5, "y": 0.5})

    rm.on_deadlock(["w", "y"])
    assert any("reduced confidence" in m for m in fh.msgs)
    assert not fh.overrides
    assert fh.replans  # both passes failed -- fell through to heavy
    assert any("falling back to a full replan" in m for m in fh.msgs)


def test_light_path_falls_back_to_heavy_when_yielder_cannot_clear():
    bottleneck = StructuralBottleneck(
        first="w",
        second="y",
        first_range=range(1, 3),
        zone_points=((1.0, 1.0), (2.0, 2.0)),
    )
    fh = FakeHooks(
        trails={"w": [(0, 0)], "y": [(9, 9)]},  # y: nothing to retreat along
        remainings={
            "w": [(0, 0), (1, 1), (2, 2), (3, 3)],
            "y": [(9, 9), (1, 1)],  # heads straight at the zone -- can't advance either
        },
        positions={"w": (0, 0), "y": (9, 9)},
        bottlenecks={frozenset(("w", "y")): bottleneck},
        plan_single=None,  # no sidestep either
    )
    rm = _manager(fh, {"w": 0.5, "y": 0.5})

    rm.on_deadlock(["w", "y"])
    assert not fh.overrides  # light path never got to override anything
    assert fh.replans  # fell straight through to the heavy (full replan) path
    assert any("falling back to a full replan" in m for m in fh.msgs)


def test_failed_yield_is_retried_before_giving_up():
    fh = FakeHooks(
        trails={"w": [(0, 0), (1, 0)], "y": [(5, 5), (4, 5), (3, 5), (2, 5)]},
        remainings={"w": [(1, 0), (2, 0)], "y": [(2, 5), (1, 5), (0, 5)]},
        positions={"w": (1, 0), "y": (2, 5)},
    )
    rm = _manager(fh, {"w": 0.5, "y": 0.5})

    rm.on_deadlock(["w", "y"])
    assert len(fh.driven) == 1  # w is winner (fewer remaining), y yields

    fh.arrive("y", success=False)
    assert len(fh.driven) == 2  # retried in place
    assert not fh.replans  # hasn't given up -- no replan on a known-dead position

    fh.arrive("y", success=False)
    assert len(fh.driven) == 3  # second retry (_MAX_YIELD_RETRIES == 2)
    assert not fh.replans

    fh.arrive("y", success=False)  # retries exhausted -> give up, proceed
    assert fh.replans
    assert any("could not be confirmed clear after retries" in m for m in fh.msgs)
