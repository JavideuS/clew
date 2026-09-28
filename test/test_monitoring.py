"""Unit tests for monitoring.py -- the pure stall/deadlock/convergence
decision logic behind coordinator_node's three execution-health checks."""

from __future__ import annotations


import pytest

from clew.monitoring import (
    ConvergenceConfig,
    DeadlockConfig,
    StallConfig,
    convergence_offset,
    convergence_streak_update,
    deadlock_group,
    stall_anchor_update,
    systematic_offset,
)

# ─── stall_anchor_update ─────────────────────────────────────────────────────


def test_stall_anchor_update_first_tick_anchors_without_stalling():
    new_anchor, is_stalled = stall_anchor_update(
        None, (1.0, 1.0), released=3, full_path=[(0, 0)] * 10, cfg=StallConfig()
    )
    assert new_anchor == ((1.0, 1.0), 3, 0)
    assert not is_stalled


def test_stall_anchor_update_streak_builds_when_unmoved():
    cfg = StallConfig(ticks=3, pos_eps_m=0.1)
    anchor = ((1.0, 1.0), 3, 0)
    anchor, stalled = stall_anchor_update(anchor, (1.0, 1.0), 3, [(0, 0)] * 10, cfg)
    assert anchor == ((1.0, 1.0), 3, 1)
    assert not stalled
    anchor, stalled = stall_anchor_update(anchor, (1.0, 1.0), 3, [(0, 0)] * 10, cfg)
    assert anchor == ((1.0, 1.0), 3, 2)
    assert not stalled
    anchor, stalled = stall_anchor_update(anchor, (1.0, 1.0), 3, [(0, 0)] * 10, cfg)
    assert anchor == ((1.0, 1.0), 3, 3)
    assert stalled  # streak reached cfg.ticks


def test_stall_anchor_update_resets_on_real_displacement():
    cfg = StallConfig(ticks=16, pos_eps_m=0.15)
    anchor = ((1.0, 1.0), 3, 10)  # a streak that was already building
    new_anchor, stalled = stall_anchor_update(
        anchor, (1.2, 1.0), released=3, full_path=[(0, 0)] * 10, cfg=cfg
    )
    assert new_anchor == ((1.2, 1.0), 3, 0)  # re-anchored at the new pos
    assert not stalled


def test_stall_anchor_update_small_jitter_does_not_reset():
    # displacement under pos_eps_m -- a single noisy sample must not cost
    # the whole streak, only a genuine move should.
    cfg = StallConfig(ticks=16, pos_eps_m=0.15)
    anchor = ((1.0, 1.0), 3, 10)
    new_anchor, stalled = stall_anchor_update(
        anchor, (1.05, 1.0), released=3, full_path=[(0, 0)] * 10, cfg=cfg
    )
    assert new_anchor == ((1.0, 1.0), 3, 11)  # anchor unchanged, streak grew
    assert not stalled


def test_stall_anchor_update_resets_on_new_release():
    # position unchanged, but more of the path was released -- real
    # progress, not a stall, even without movement yet.
    cfg = StallConfig(ticks=16, pos_eps_m=0.15)
    anchor = ((1.0, 1.0), 3, 10)
    new_anchor, stalled = stall_anchor_update(
        anchor, (1.0, 1.0), released=5, full_path=[(0, 0)] * 10, cfg=cfg
    )
    assert new_anchor == ((1.0, 1.0), 5, 0)
    assert not stalled


def test_stall_anchor_update_clears_at_goal():
    cfg = StallConfig(ticks=16, pos_eps_m=0.15)
    full_path = [(0, 0), (1, 0), (2, 0)]
    anchor = ((2.0, 0.0), 3, 12)
    new_anchor, stalled = stall_anchor_update(
        anchor, (2.0, 0.0), released=3, full_path=full_path, cfg=cfg
    )
    assert new_anchor is None
    assert not stalled


# ─── deadlock_group ───────────────────────────────────────────────────────────


def test_deadlock_group_empty_when_nothing_close_or_chained():
    group, collisions, chains = deadlock_group(
        stuck_ids=["a", "b"],
        positions={"a": (0.0, 0.0), "b": (10.0, 10.0)},
        effective_radii={"a": 0.5, "b": 0.5},
        pending_blockers={"a": [], "b": []},
        cfg=DeadlockConfig(proximity_factor=1.25),
    )
    assert group == frozenset()
    assert collisions == []
    assert chains == []


def test_deadlock_group_finds_close_stuck_pair():
    group, collisions, chains = deadlock_group(
        stuck_ids=["a", "b"],
        positions={"a": (0.0, 0.0), "b": (1.0, 0.0)},
        effective_radii={"a": 0.5, "b": 0.5},
        pending_blockers={"a": [], "b": []},
        cfg=DeadlockConfig(proximity_factor=1.25),
    )
    assert group == frozenset({"a", "b"})
    assert collisions == [("a", "b", 1.0, 1.0)]
    assert chains == []


def test_deadlock_group_finds_chain_through_a_non_stuck_robot():
    # c is release-gated behind stuck a, even though c itself isn't stuck.
    group, collisions, chains = deadlock_group(
        stuck_ids=["a"],
        positions={"a": (0.0, 0.0)},
        effective_radii={"a": 0.5},
        pending_blockers={"a": [], "c": [("a", 5, 2)]},
        cfg=DeadlockConfig(proximity_factor=1.25),
    )
    assert group == frozenset({"a", "c"})
    assert collisions == []
    assert chains == [("c", "a", 5, 2)]


# ─── convergence_offset ───────────────────────────────────────────────────────


def test_convergence_offset_within_tolerance_is_none():
    cfg = ConvergenceConfig(pos_tol_m=0.35, yaw_tol_rad=0.25)
    dx, dy, dyaw, blocker = convergence_offset(
        (0.1, 0.0), tf_yaw=0.1, seed=(0.0, 0.0, 0.0), cfg=cfg
    )
    assert blocker is None
    assert dx == pytest.approx(0.1)
    assert dyaw == pytest.approx(0.1)


def test_convergence_offset_position_out_of_tolerance():
    cfg = ConvergenceConfig(pos_tol_m=0.35, yaw_tol_rad=0.25)
    _, _, _, blocker = convergence_offset(
        (1.0, 0.0), tf_yaw=0.0, seed=(0.0, 0.0, 0.0), cfg=cfg
    )
    assert blocker is not None
    assert "1.00 m" in blocker


def test_convergence_offset_yaw_out_of_tolerance():
    cfg = ConvergenceConfig(pos_tol_m=0.35, yaw_tol_rad=0.25)
    _, _, _, blocker = convergence_offset(
        (0.0, 0.0), tf_yaw=1.0, seed=(0.0, 0.0, 0.0), cfg=cfg
    )
    assert blocker is not None
    assert "yaw" in blocker


# ─── convergence_streak_update ────────────────────────────────────────────────


def test_convergence_streak_builds_to_stable_then_converges():
    cfg = ConvergenceConfig(stable_polls=2)
    streak, converged = convergence_streak_update(0, blocked=False, cfg=cfg)
    assert (streak, converged) == (1, False)
    streak, converged = convergence_streak_update(streak, blocked=False, cfg=cfg)
    assert (streak, converged) == (2, True)


def test_convergence_streak_resets_when_blocked():
    cfg = ConvergenceConfig(stable_polls=2)
    streak, converged = convergence_streak_update(1, blocked=True, cfg=cfg)
    assert (streak, converged) == (0, False)


# ─── systematic_offset ─────────────────────────────────────────────────────────


def test_systematic_offset_false_for_fewer_than_two_robots():
    cfg = ConvergenceConfig(pos_tol_m=0.35, systematic_agreement_tol_m=0.30)
    assert not systematic_offset({"a": (1.0, 1.0)}, cfg)


def test_systematic_offset_true_when_offsets_agree_and_exceed_tolerance():
    cfg = ConvergenceConfig(pos_tol_m=0.35, systematic_agreement_tol_m=0.30)
    means = {"a": (0.5, 0.5), "b": (0.52, 0.48)}
    assert systematic_offset(means, cfg)


def test_systematic_offset_false_when_offsets_scatter():
    cfg = ConvergenceConfig(pos_tol_m=0.35, systematic_agreement_tol_m=0.30)
    means = {"a": (0.5, 0.5), "b": (-0.5, 0.2)}
    assert not systematic_offset(means, cfg)


def test_systematic_offset_false_when_agreement_is_below_pos_tolerance():
    # offsets agree, but the shared offset itself is inside tolerance --
    # not a frame error, just normal convergence noise.
    cfg = ConvergenceConfig(pos_tol_m=0.35, systematic_agreement_tol_m=0.30)
    means = {"a": (0.05, 0.05), "b": (0.06, 0.04)}
    assert not systematic_offset(means, cfg)
