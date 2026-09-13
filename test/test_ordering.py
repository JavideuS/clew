"""Unit tests for ordering.py -- conflict-zone release-gating."""

from __future__ import annotations

from fleet_coordinator.ordering import ReleaseSchedule
from fleet_coordinator.spooky_client import RobotPlan


def _plan(robot_id: str, path: list[tuple[float, float]]) -> RobotPlan:
    return RobotPlan(robot_id=robot_id, path=path, coordinate_format="world")


def test_no_conflict_fully_released_immediately():
    # Two robots whose paths never come close -- nothing should gate them.
    plans = {
        "a": _plan("a", [(0, 0), (1, 0), (2, 0)]),
        "b": _plan("b", [(0, 10), (1, 10), (2, 10)]),
    }
    schedule = ReleaseSchedule.from_fleet_plan(plans, {"a": 0.3, "b": 0.3})

    assert schedule.released_path("a") == [(0, 0), (1, 0), (2, 0)]
    assert schedule.released_path("b") == [(0, 10), (1, 10), (2, 10)]
    assert schedule.pending_blockers("a") == []
    assert schedule.pending_blockers("b") == []


def test_earlier_index_robot_goes_first_other_waits_until_it_pulls_clear():
    # a reaches (5, 0) at index 1; b reaches the same point at index 2 --
    # a's symbolic index is earlier, so it goes first and b waits. b is not
    # released past the zone until a has pulled a full clearance distance
    # (>= combined_radius = 1.0) clear of the shared point, which for a
    # only happens at index 2 (6.1, 0) -- reaching index 1 isn't enough.
    # a[1]->a[2] is a real-grid-scale step (1.1 m, just over clear_distance),
    # not a huge jump, so this reads as an ordinary brief crossing rather
    # than a structural bottleneck.
    plans = {
        "a": _plan("a", [(0, 0), (5, 0), (6.1, 0)]),
        "b": _plan("b", [(5, 5), (5, 2), (5, 0), (5, -5)]),
    }
    schedule = ReleaseSchedule.from_fleet_plan(plans, {"a": 0.5, "b": 0.5})
    assert schedule.structural_bottleneck("a", "b") is None

    # a is never gated -- its whole path releases immediately.
    assert schedule.released_path("a") == [(0, 0), (5, 0), (6.1, 0)]

    # b is held at the zone mouth (index 1), one short of the conflict point.
    assert schedule.released_path("b") == [(5, 5), (5, 2)]
    assert schedule.pending_blockers("b") == [("a", 2, -1)]

    # a merely reaching the shared cell is not enough to let b in.
    schedule.report_progress("a", 1)
    assert schedule.released_path("b") == [(5, 5), (5, 2)]

    # a a step past it (clear by >= 1.0) -- b's whole remaining path releases.
    schedule.report_progress("a", 2)
    assert schedule.released_path("b") == [(5, 5), (5, 2), (5, 0), (5, -5)]
    assert schedule.pending_blockers("b") == []


def test_whole_shared_band_is_one_gate_not_per_cell():
    # Near-mirror routes down a shared corridor along y=0: a runs it
    # left-to-right, b comes in from above and runs it right-to-left. The
    # old per-cell gate handed the shared cells over one at a time (b
    # parking a cell behind a the whole way); zone gating holds b entirely
    # off the corridor until a is clear of all of it.
    corridor_a = [(float(x), 0.0) for x in range(7)]  # (0,0)..(6,0)
    b_path = [(6.0, 3.0), (6.0, 2.0)] + [(float(x), 0.0) for x in range(6, -1, -1)]
    plans = {"a": _plan("a", corridor_a), "b": _plan("b", b_path)}
    schedule = ReleaseSchedule.from_fleet_plan(plans, {"a": 0.5, "b": 0.5})

    # a enters the shared zone at step 0, b only at step 2 -> a goes first,
    # unobstructed.
    assert schedule.released_path("a") == corridor_a

    # b is held at the zone mouth: its lead-in (indices 0-1) releases, the
    # corridor (index 2 on) does not.
    assert schedule.released_path("b") == [(6.0, 3.0), (6.0, 2.0)]

    # Partial progress by a still doesn't crack the gate open cell by cell.
    schedule.report_progress("a", 4)
    assert schedule.released_path("b") == [(6.0, 3.0), (6.0, 2.0)]

    # Only once a has finished (pulled clear of the whole band) does b go.
    schedule.report_progress("a", 6)
    assert schedule.released_path("b") == b_path


def test_two_disjoint_zones_keep_one_consistent_order():
    # a and b cross twice, far apart. Order is fixed for the pair from the
    # earliest zone (a first), and applied to both -- b never leapfrogs a
    # into a precedence cycle. a's point right after the first crossing,
    # (1, 1.3), is only just past clear_distance (1.2) from it -- a real
    # crossing-and-clear, not the huge jump the original (2, 5) was, which
    # (being ~5x combined_radius away) misread as a structural bottleneck.
    plans = {
        "a": _plan("a", [(0, 0), (1, 0), (1.0, 1.3), (3, 10), (4, 10), (5, 10)]),
        "b": _plan("b", [(1, 0), (1, 5), (5, 10), (9, 10), (9, 5), (9, 0)]),
    }
    schedule = ReleaseSchedule.from_fleet_plan(plans, {"a": 0.6, "b": 0.6})
    assert schedule.structural_bottleneck("a", "b") is None

    assert schedule.released_path("a") == [
        (0, 0), (1, 0), (1.0, 1.3), (3, 10), (4, 10), (5, 10),
    ]

    # b is gated before the first shared point (index 0, near a's (1,0)).
    assert schedule.released_path("b") == []
    blockers = schedule.pending_blockers("b")
    assert blockers and all(other == "a" for other, _, _ in blockers)

    schedule.report_progress("a", 5)  # a finishes everything
    assert schedule.released_path("b") == [
        (1, 0), (1, 5), (5, 10), (9, 10), (9, 5), (9, 0),
    ]
    assert schedule.pending_blockers("b") == []


def test_report_progress_is_monotonic():
    plans = {
        "a": _plan("a", [(0, 0), (1, 0)]),
        "b": _plan("b", [(1, 0), (2, 0)]),
    }
    schedule = ReleaseSchedule.from_fleet_plan(plans, {"a": 0.5, "b": 0.5})
    schedule.report_progress("a", 1)
    schedule.report_progress("a", 0)  # stale/out-of-order report -- must not un-clear
    assert schedule.released_path("b") == [(1, 0), (2, 0)]


def test_mixed_coordinate_format_rejected():
    plans = {
        "a": RobotPlan(robot_id="a", path=[(0, 0), (1, 0)], coordinate_format="world"),
        "b": RobotPlan(robot_id="b", path=[(0, 0), (1, 0)], coordinate_format="cartesian"),
    }
    try:
        ReleaseSchedule.from_fleet_plan(plans, {"a": 0.5, "b": 0.5})
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "mixed coordinate_formats" in str(exc)


def test_same_step_conflict_tie_breaks_deterministically(caplog):
    # Both robots at the same point, same symbolic index -- a planning-side
    # anomaly Spooky's own penalties should prevent, but must not deadlock.
    plans = {
        "a": _plan("a", [(0, 0), (5, 5)]),
        "b": _plan("b", [(9, 9), (5, 5)]),
    }
    schedule = ReleaseSchedule.from_fleet_plan(plans, {"a": 0.5, "b": 0.5})

    # deterministic tie-break: lower robot id ("a") goes first
    assert schedule.released_path("a") == [(0, 0), (5, 5)]
    assert schedule.released_path("b") == [(9, 9)]
    assert any("tie-breaking by robot id" in rec.message for rec in caplog.records)

    schedule.report_progress("a", 1)
    assert schedule.released_path("b") == [(9, 9), (5, 5)]


def test_structural_swap_gates_both_robots_at_the_mouth(caplog):
    # Near-mirror swap: a goes up-left of the diagonal, b down-right, ~1
    # cell apart the whole way. a's route hugs b's start, so b can't hold
    # anywhere clear before the shared stretch -> BOTH get gated at their
    # zone mouths (not b alone parked at its start forever).
    import logging

    a = [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3), (3, 3), (3, 4), (4, 4)]
    b = [(4, 4), (4, 3), (3, 3), (3, 2), (2, 2), (2, 1), (1, 1), (1, 0), (0, 0)]
    plans = {"a": _plan("a", a), "b": _plan("b", b)}

    with caplog.at_level(logging.WARNING):
        schedule = ReleaseSchedule.from_fleet_plan(plans, {"a": 0.6, "b": 0.6})

    assert any("structural bottleneck" in r.message for r in caplog.records)
    assert not any(
        "will deadlock, each gated behind the next" in r.message
        for r in caplog.records
    )

    # neither robot runs its whole path; both are held partway with an
    # unmet blocker -- the symmetric mid-route state the stall detector
    # hands to recovery.
    rel_a, rel_b = schedule.released_path("a"), schedule.released_path("b")
    assert 1 <= len(rel_a) < len(a)
    assert 1 <= len(rel_b) < len(b)
    assert schedule.pending_blockers("a")
    assert schedule.pending_blockers("b")


def test_structural_trigger_catches_a_hold_pose_that_still_waits_almost_the_whole_trip(
    caplog,
):
    # Real near-mirror swap paths (from a live-run bug: two robots swapping
    # start/goal through one grid corridor). At this radius, sim2 finds a
    # hold pose only 2 steps in (gate=2, not 0) -- the original gate==0
    # check alone would miss this and release sim1 in full, exactly the
    # "0 parallelism" symptom that motivated the broadened trigger.
    import logging

    sim1 = [
        (0.0, 0.0), (0.225, -0.175), (0.625, -0.175), (0.625, 0.225),
        (1.025, 0.225), (1.025, 0.625), (1.025, 1.025), (1.025, 1.425),
        (1.425, 1.425), (1.425, 1.825), (1.825, 1.825), (1.825, 2.225),
        (2.225, 2.225), (2.625, 2.225), (2.625, 2.625), (3.025, 2.625),
        (3.025, 3.025), (3.025, 3.425), (3.025, 3.825), (3.025, 4.225),
        (3.425, 4.225), (3.825, 4.225), (3.825, 4.625), (4.225, 4.625),
        (4.625, 4.625), (4.625, 5.025), (5.0, 5.0),
    ]
    sim2 = [
        (5.025, 5.025), (4.625, 5.025), (4.225, 5.025), (3.825, 5.025),
        (3.425, 5.025), (3.425, 4.625), (3.425, 4.225), (3.425, 3.825),
        (3.425, 3.425), (3.025, 3.425), (2.625, 3.425), (2.625, 3.025),
        (2.625, 2.625), (2.225, 2.625), (2.225, 2.225), (2.225, 1.825),
        (1.825, 1.825), (1.825, 1.425), (1.425, 1.425), (1.425, 1.025),
        (1.425, 0.625), (1.425, 0.225), (1.425, -0.175), (1.025, -0.175),
        (0.625, -0.175), (0.225, -0.175), (-0.175, -0.175),
    ]
    plans = {"a": _plan("a", sim1), "b": _plan("b", sim2)}

    with caplog.at_level(logging.WARNING):
        schedule = ReleaseSchedule.from_fleet_plan(plans, {"a": 0.5, "b": 0.5})

    assert any("structural bottleneck" in r.message for r in caplog.records)
    assert schedule.structural_bottleneck("a", "b") is not None

    rel_a, rel_b = schedule.released_path("a"), schedule.released_path("b")
    assert 1 <= len(rel_a) < len(sim1)
    assert 1 <= len(rel_b) < len(sim2)
    # symmetric -- neither robot got the free pass the narrower check gave
    # ranger_sim1 in the live bug (released 27/27 while sim2 sat at 2/27)
    assert abs(len(rel_a) - len(rel_b)) <= 1


def test_release_override_drops_only_the_named_constraints():
    # Same fixture as test_structural_swap_gates_both_robots_at_the_mouth --
    # known to trigger the gate==0 structural path.
    a = [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3), (3, 3), (3, 4), (4, 4)]
    b = [(4, 4), (4, 3), (3, 3), (3, 2), (2, 2), (2, 1), (1, 1), (1, 0), (0, 0)]
    plans = {"a": _plan("a", a), "b": _plan("b", b)}
    schedule = ReleaseSchedule.from_fleet_plan(plans, {"a": 0.6, "b": 0.6})

    bottleneck = schedule.structural_bottleneck("a", "b")
    assert bottleneck is not None

    # a is gated -- confirm it, then override exactly the recorded range
    assert len(schedule.released_path("a")) < len(a)
    schedule.release_override(bottleneck.first, bottleneck.first_range)
    assert schedule.released_path(bottleneck.first) == a

    # b's own (untouched) constraint still holds it back
    assert len(schedule.released_path(bottleneck.second)) < len(b)


def test_precedence_cycle_is_detected_and_logged(caplog):
    # Three robots, three well-separated crossings, each ordered so the
    # "waits for" edges form a loop: a<b (b waits for a), b<c (c waits for
    # b), c<a (a waits for c) -> a -> c -> b -> a. A genuine deadlock;
    # from_fleet_plan can't fix it but must flag it loudly.
    import logging

    # Only the three (x, 0) points are shared crossings; the three
    # crossings (100 apart) and each robot's post-crossing point (1.1 --
    # just past clear_distance=1.0, a real-grid-scale step) are far enough
    # apart that nothing else registers as close, without a huge jump that
    # would misread as a structural bottleneck.
    plans = {
        # crosses at (0,0) on step 1, at (200,0) on step 3
        "a": _plan("a", [(0, -2.0), (0, 0), (1.1, 0), (200, 0), (200, 2.0), (200, 4.0)]),
        # crosses at (100,0) on step 1, at (0,0) on step 3
        "b": _plan("b", [(100, -2.0), (100, 0), (101.1, 0), (0, 0), (0, 2.0), (0, 4.0)]),
        # crosses at (200,0) on step 1, at (100,0) on step 3
        "c": _plan("c", [(200, -2.0), (200, 0), (201.1, 0), (100, 0), (100, 2.0), (100, 4.0)]),
    }
    with caplog.at_level(logging.ERROR):
        ReleaseSchedule.from_fleet_plan(plans, {"a": 0.5, "b": 0.5, "c": 0.5})
    assert any("cyclic" in rec.message.lower() for rec in caplog.records)
