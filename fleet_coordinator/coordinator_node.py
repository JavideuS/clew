"""Top-level rclpy node: wires spooky_client, ordering, and dispatch
together.

mission intake (robot.py), global planning (spooky_client.py),
release-gating (ordering.py), and dispatch (dispatch.py, straight into
nav2's FollowPath

Full loop:

1. Load per-robot start/goal pairs, via a mission YAML file
   (Fleet.from_yaml). Revisit if/when goals need to come from a live
   source instead of a static file (README.md §7).
2. spooky_client.plan_fleet(...) once for the whole fleet.
3. Build an ordering.ReleaseSchedule from the resulting paths.
4. On a timer: for each robot, dispatch.RobotDispatcher.publish_released_path(
   release_schedule.released_path(robot_id)). No-op once nothing new
   has been released since the last tick, so this is safe to call on every
   tick regardless of whether anything actually changed.
5. RobotDispatcher's own pose subscription reports progress back via
   release_schedule.report_progress(...) as robots move, independently of
   the timer above.
"""

from __future__ import annotations

from dataclasses import fields, replace
from typing import TypeVar

import rclpy
from rclpy.node import Node

from .convergence import ConvergenceGate
from .dispatch import DispatchConfig, RobotDispatcher
from .geometry import distance
from .monitoring import (
    ConvergenceConfig,
    DeadlockConfig,
    StallAnchor,
    StallConfig,
    deadlock_group,
    stall_anchor_update,
)
from .ordering import OrderingConfig, ReleaseSchedule
from .recovery import RecoveryConfig, RecoveryHooks, RecoveryManager
from .robot import Fleet, Pose2D, Robot
from .spooky_client import SpookyPlanError, SpookySettings, plan_fleet

# Dispatch tick: only needs to notice a new release and republish -- not
# a control-loop rate (the local planner runs its own, much faster).
_DISPATCH_TICK_PERIOD_S = 0.5

_DC = TypeVar("_DC")


def declare_dataclass_parameters(node: Node, prefix: str, defaults) -> None:
    """declare_parameter(f"{prefix}.{field}", default) for every field of
    the dataclass instance `defaults` -- every tunable in this file is
    grouped into a config dataclass (SpookySettings, StallConfig, ...)
    *and* exposed as a ROS2 parameter under the same name, so this and
    load_dataclass_parameters are the only two places that pairing is
    spelled out. Add a field to a config dataclass and it's declared and
    loaded automatically.
    """
    for f in fields(defaults):
        node.declare_parameter(f"{prefix}.{f.name}", getattr(defaults, f.name))


def load_dataclass_parameters(node: Node, prefix: str, cls: type[_DC]) -> _DC:
    """The inverse of declare_dataclass_parameters: build a `cls` instance
    from whatever value each `{prefix}.{field}` parameter currently holds
    (a launch-file/CLI override, or the declared default). `cls` must
    already have had declare_dataclass_parameters(node, prefix, cls())
    called for it -- get_parameter raises otherwise.
    """
    kwargs = {
        f.name: node.get_parameter(f"{prefix}.{f.name}").value for f in fields(cls)
    }
    return cls(**kwargs)


class CoordinatorNode(Node):
    def __init__(self) -> None:
        super().__init__("fleet_coordinator")

        # Set for real in _plan_fleet; referenced defensively before then.
        self.recovery: RecoveryManager | None = None
        self._recovery_halted = False

        self.declare_parameter("mission_file", "")
        # Every field of these five config dataclasses becomes a
        # same-named ROS2 parameter -- see declare_dataclass_parameters.
        for prefix, defaults in [
            ("spooky", SpookySettings()),
            ("stall", StallConfig()),
            ("deadlock", DeadlockConfig()),
            ("convergence", ConvergenceConfig()),
            ("ordering", OrderingConfig()),
            ("dispatch", DispatchConfig()),
            ("recovery", RecoveryConfig()),
        ]:
            declare_dataclass_parameters(self, prefix, defaults)

        # AMCL seeding: publish each robot's initial pose, then gate
        # planning on convergence. Opt-in (defaults off) -- see
        # convergence.ConvergenceGate's own docstring for what "publish"
        # must mean here: whatever pose you seed with must be a trustworthy
        # measurement, not an assumption. The mission file's declared
        # start is only trustworthy as that measurement in sim (you
        # control both); a real deployment needs a real source (operator
        # confirmation, a docking station, ...) before enabling this.
        self.declare_parameter("initial_pose.publish", False)
        self.declare_parameter("initial_pose.topic_template", "/{robot_id}/initialpose")
        # Per-robot TF tree: nav2_bringup with use_namespace remaps /tf and
        # /tf_static into the robot's namespace, so each robot's transforms
        # live on /{robot_id}/tf(+_static) with bare frame ids inside.
        # Override base_frame_template to "{robot_id}/base_link" for setups
        # that prefix frame ids instead.
        self.declare_parameter("initial_pose.tf_topic_template", "/{robot_id}/tf")
        self.declare_parameter(
            "initial_pose.tf_static_topic_template", "/{robot_id}/tf_static"
        )
        self.declare_parameter("initial_pose.map_frame", "map")
        self.declare_parameter("initial_pose.base_frame_template", "base_link")

        mission_file = self.get_parameter("mission_file").value
        if not mission_file:
            self.get_logger().error(
                "coordinator_node: no mission_file parameter set, nothing to plan"
            )
            self.fleet = None
            return

        self.fleet = Fleet.from_yaml(mission_file)
        self.get_logger().info(
            f"coordinator_node: loaded {len(self.fleet)} robot(s) from {mission_file!r}"
        )

        self.spooky_settings = load_dataclass_parameters(self, "spooky", SpookySettings)
        self._stall_cfg = load_dataclass_parameters(self, "stall", StallConfig)
        self._deadlock_cfg = load_dataclass_parameters(self, "deadlock", DeadlockConfig)
        self._convergence_cfg = load_dataclass_parameters(
            self, "convergence", ConvergenceConfig
        )
        self._ordering_cfg = load_dataclass_parameters(self, "ordering", OrderingConfig)
        self._dispatch_cfg = load_dataclass_parameters(self, "dispatch", DispatchConfig)
        self._recovery_cfg = load_dataclass_parameters(self, "recovery", RecoveryConfig)

        if self.get_parameter("initial_pose.publish").value:
            self._convergence_gate = ConvergenceGate(
                self,
                self.fleet,
                self._convergence_cfg,
                on_converged=self._start_planning,
                topic_template=self.get_parameter("initial_pose.topic_template").value,
                tf_topic_template=self.get_parameter(
                    "initial_pose.tf_topic_template"
                ).value,
                tf_static_topic_template=self.get_parameter(
                    "initial_pose.tf_static_topic_template"
                ).value,
                map_frame=self.get_parameter("initial_pose.map_frame").value,
                base_frame_template=self.get_parameter(
                    "initial_pose.base_frame_template"
                ).value,
            )
            self._convergence_gate.start()  # calls _start_planning() once converged
        else:
            self._plan_fleet()

    def _start_planning(self) -> None:
        # Guard: the all-converged branch and the deadline branch can fire
        # in the same poll, and a poll callback may already be queued when
        # the timer is cancelled. Plan exactly once.
        if getattr(self, "_planning_started", False):
            return
        self._planning_started = True

        # No separate post-convergence settle wait: the gate already required
        # a stable map->base_link for cfg.stable_polls polls (~2 s of them
        # by default), so AMCL has processed several scans by now, and
        # _plan_fleet() itself
        # is a blocking Spooky call (~2 s) that touches neither TF nor the
        # local costmap -- any remaining costmap settling overlaps it for
        # free. Dispatch only starts on the first _dispatch_tick after this.

        # Plan from where each robot actually settled, not the mission
        # file's declared start, so a robot that AMCL correctly
        # scan-matched away from an inaccurate declared start would
        # otherwise plan from a point nav2 will immediately disagree with.
        # Robots that never settled (timed out) keep their declared start
        # -- see _plan_fleet() -> planning anyway with what we have.
        gate = getattr(self, "_convergence_gate", None)
        if gate is not None:
            for robot_id, (x, y, yaw) in gate.settled_poses.items():
                robot = self.fleet.robots[robot_id]
                declared = robot.start
                off = distance((x, y), (declared.x, declared.y))
                if off > self._convergence_cfg.pos_tol_m:
                    self.get_logger().warning(
                        f"coordinator_node: {robot_id}'s declared start "
                        f"({declared.x}, {declared.y}) was {off:.2f} m from "
                        f"where it actually settled ({x:.2f}, {y:.2f}) -- "
                        "planning from the settled pose instead"
                    )
                robot.start = Pose2D(x, y, yaw)

        self._plan_fleet()

    def _plan_fleet(self) -> None:
        try:
            plan = plan_fleet(self.fleet, self.spooky_settings)
        except SpookyPlanError as exc:
            self.get_logger().error(f"coordinator_node: global planning failed: {exc}")
            return

        self.get_logger().info(
            f"coordinator_node: got a plan for {len(plan.robot_plans)} "
            f"robot(s), cost={plan.cost}"
        )

        effective_radii = {robot.id: robot.effective_radius for robot in self.fleet}
        self._effective_radii = effective_radii
        self._reported_deadlocks: set[frozenset[str]] = set()
        self._recovery_halted = False
        # Set once every dispatcher reports is_complete() -- see
        # _dispatch_tick.
        self._mission_done = False
        # Per-robot stall-detection anchor -- see monitoring.stall_anchor_update.
        # Cleared on every (re)plan.
        self._stall_anchor: dict[str, StallAnchor] = {}
        self.release_schedule = self._build_release_schedule(plan)

        self.dispatchers: dict[str, RobotDispatcher] = {}
        for robot_id, robot_plan in plan.robot_plans.items():
            robot = self.fleet.robots[robot_id]
            self.dispatchers[robot_id] = RobotDispatcher(
                robot_id,
                self,
                full_path=robot_plan.path,
                start_theta=robot.start.theta,
                goal_theta=robot.goal.theta,
                # Closure captures robot_id by value via the default arg --
                # a plain lambda referencing the loop variable directly
                # would have every dispatcher's callback report progress
                # for whichever robot_id the loop landed on *last*.
                on_progress=lambda idx,
                rid=robot_id: self.release_schedule.report_progress(rid, idx),
                cfg=self._dispatch_cfg,
                on_stuck=self._on_robot_stuck,
            )
            released = self.release_schedule.released_path(robot_id)
            self.get_logger().info(
                f"coordinator_node: {robot_id} initially released "
                f"{len(released)}/{len(robot_plan.path)} path point(s)"
            )

        self.recovery = RecoveryManager(
            priorities={r.id: r.priority for r in self.fleet},
            effective_radii=effective_radii,
            hooks=RecoveryHooks(
                position_of=self._robot_position,
                trail_of=lambda rid: self.dispatchers[rid].trail(),
                remaining_of=lambda rid: self.dispatchers[rid].remaining(),
                drive_oneoff=lambda rid, poses, on_done, reason: (
                    self.dispatchers[rid].send_oneoff_path(poses, on_done)
                ),
                hold=lambda rid: self.dispatchers[rid].pause(),
                resume=lambda rid: self.dispatchers[rid].resume(),
                structural_bottleneck=(
                    lambda a, b: self.release_schedule.structural_bottleneck(a, b)
                ),
                release_override=(
                    lambda rid, indices: self.release_schedule.release_override(
                        rid, indices
                    )
                ),
                plan_single=self._recovery_plan_single,
                replan=self._replan_from_current,
                fail=self._recovery_failed,
                log=lambda m: self.get_logger().info(f"coordinator_node: {m}"),
                warn=lambda m: self.get_logger().warning(f"coordinator_node: {m}"),
            ),
            clearance_factor=self._ordering_cfg.clearance_factor,
            cfg=self._recovery_cfg,
        )

        self.create_timer(_DISPATCH_TICK_PERIOD_S, self._dispatch_tick)

    def _build_release_schedule(self, plan) -> ReleaseSchedule:
        return ReleaseSchedule.from_fleet_plan(
            plan.robot_plans, self._effective_radii, self._ordering_cfg
        )

    def _dispatch_tick(self) -> None:
        if self._recovery_halted:
            return
        for robot_id, dispatcher in self.dispatchers.items():
            dispatcher.publish_released_path(
                self.release_schedule.released_path(robot_id)
            )
        self._check_fleet_stall()
        self._check_mission_done()

    def _check_mission_done(self) -> None:
        """Shut this node down once every robot has actually finished its
        full path -- not just reached the end of its currently-released
        prefix (RobotDispatcher.is_complete() already tells those apart).

        Meant for it close once mission finishes. Allows better integration
        and usage for planning iteratively on real time. Meant for argOS world
        orchestration and future task-allocation work.
        """
        if self._mission_done or not self.dispatchers:
            return
        if not all(d.is_complete() for d in self.dispatchers.values()):
            return
        self._mission_done = True
        self.get_logger().info(
            "coordinator_node: mission complete -- every robot reached its "
            "goal, shutting down"
        )
        rclpy.shutdown()

    def _check_fleet_stall(self) -> None:
        """Catch a deadlock that never produces a controller abort: robots
        stopped clean at a mutual release gate (structural bottleneck) and
        are just sitting there. If >=2 not-yet-home robots have made no
        progress -- no movement, no newly released path -- for
        cfg.ticks dispatch ticks and at least one is release-gated, hand
        them to recovery. See monitoring.stall_anchor_update for the
        per-robot streak logic.
        """
        if (
            self.recovery is None
            or self._recovery_halted
            or self.recovery.is_recovering()
        ):
            return

        stalled: list[str] = []
        for robot_id, dispatcher in self.dispatchers.items():
            full_path = dispatcher.full_path()
            released = len(self.release_schedule.released_path(robot_id))
            pos = self._robot_position(robot_id)
            old_anchor = self._stall_anchor.get(robot_id)

            new_anchor, is_stalled = stall_anchor_update(
                old_anchor, pos, released, full_path, self._stall_cfg
            )

            # Only worth logging a reset when it interrupts a streak that
            # was actually building (a robot genuinely still driving
            # resets every tick, which would spam this) -- diagnostic for
            # tracking down what's delaying detection beyond cfg.ticks.
            if (
                new_anchor is not None
                and new_anchor[2] == 0
                and old_anchor is not None
                and old_anchor[2] > 0
            ):
                disp = distance(pos, old_anchor[0])
                self.get_logger().debug(
                    f"coordinator_node: {robot_id}'s stall streak reset "
                    f"after {old_anchor[2]} tick(s) -- displacement "
                    f"{disp:.3f}m from anchor, released {old_anchor[1]}->{released}"
                )

            if new_anchor is None:
                self._stall_anchor.pop(robot_id, None)
            else:
                self._stall_anchor[robot_id] = new_anchor
            if is_stalled:
                stalled.append(robot_id)

        if len(stalled) < 2:
            return

        gated = [r for r in stalled if self.release_schedule.pending_blockers(r)]
        group = gated if len(gated) >= 2 else stalled
        stall_s = self._stall_cfg.ticks * _DISPATCH_TICK_PERIOD_S
        self.get_logger().error(
            f"coordinator_node: FLEET STALL -- {sorted(group)} made no progress "
            f"for ~{stall_s:.0f}s at a release gate; handing to coordinated "
            "recovery"
        )
        self.recovery.on_deadlock(sorted(group))

    # ── recovery wiring ───────────────────────────────────────────────────

    def _recovery_plan_single(
        self, robot_id: str, start: tuple[float, float], goal: tuple[float, float]
    ) -> list[tuple[float, float]] | None:
        """Single-robot Spooky plan for a recovery sidestep. Returns the
        path points, or None if Spooky can't route it (no free cell there).

        Uses RecoveryConfig.sidestep_timeout_s, not spooky.timeout_s
        """
        base = self.fleet.robots[robot_id]
        one = Robot(
            id=robot_id,
            start=Pose2D(start[0], start[1], base.start.theta),
            goal=Pose2D(goal[0], goal[1], base.goal.theta),
            start_time=base.start_time,
            priority=base.priority,
            robot_radius=base.robot_radius,
            inflation=base.inflation,
            coordinate_format=base.coordinate_format,
        )
        probe_settings = replace(
            self.spooky_settings, timeout_s=self._recovery_cfg.sidestep_timeout_s
        )
        try:
            plan = plan_fleet(Fleet(robots={robot_id: one}), probe_settings)
        except SpookyPlanError as exc:
            self.get_logger().warning(
                f"coordinator_node: recovery sidestep plan for {robot_id} "
                f"failed: {exc}"
            )
            return None
        return plan.robot_plans[robot_id].path

    def _replan_from_current(
        self,
        starts: dict[str, tuple[float, float]],
        pinned_priority: dict[str, float] | None,
    ) -> bool:
        """Full-fleet CBS from every robot's current pose, rebuild the
        ReleaseSchedule, swap each dispatcher onto its new path, resume
        normal dispatch. Returns False if Spooky can't solve it.
        """
        pinned = pinned_priority or {}
        robots = {}
        for robot in self.fleet:
            sx, sy = starts.get(robot.id, (robot.start.x, robot.start.y))
            robots[robot.id] = Robot(
                id=robot.id,
                start=Pose2D(sx, sy, robot.start.theta),
                goal=robot.goal,
                start_time=robot.start_time,
                priority=pinned.get(robot.id, robot.priority),
                robot_radius=robot.robot_radius,
                inflation=robot.inflation,
                coordinate_format=robot.coordinate_format,
            )
        try:
            plan = plan_fleet(Fleet(robots=robots), self.spooky_settings)
        except SpookyPlanError as exc:
            self.get_logger().error(f"coordinator_node: recovery replan failed: {exc}")
            return False

        self.get_logger().info(
            f"coordinator_node: recovery replan got a plan, cost={plan.cost}"
        )
        self.release_schedule = self._build_release_schedule(plan)
        self._reported_deadlocks.clear()
        self._stall_anchor.clear()
        for robot_id, robot_plan in plan.robot_plans.items():
            self.dispatchers[robot_id].update_path(robot_plan.path)
        return True

    def _recovery_failed(self, reason: str) -> None:
        self.get_logger().error(
            f"coordinator_node: RECOVERY FAILED -- {reason}. Halting all "
            "robots; operator intervention needed."
        )
        self._recovery_halted = True
        for dispatcher in self.dispatchers.values():
            dispatcher.pause()

    def _on_robot_stuck(self, robot_id: str, abort_count: int) -> None:
        """Called by a RobotDispatcher once its FollowPath has aborted
        max_consecutive_aborts times in a row on the same released prefix
        and it has stopped re-sending. Report *why* for this one robot,
        then check whether the fleet as a whole is now deadlocked.
        """
        blockers = self.release_schedule.pending_blockers(robot_id)
        if blockers:
            detail = "; ".join(
                f"{other} to reach index {need} (cleared {have} so far)"
                for other, need, have in blockers
            )
            self.get_logger().error(
                f"coordinator_node: {robot_id} stuck after {abort_count} "
                f"controller aborts -- release-gated, waiting on {detail}."
            )
        else:
            self.get_logger().error(
                f"coordinator_node: {robot_id} stuck after {abort_count} "
                "controller aborts, not release-gated (its whole path is "
                "released)."
            )
        self._check_fleet_deadlock()

    def _robot_position(self, robot_id: str) -> tuple[float, float]:
        """Best available (x, y) for a robot: its last seen amcl_pose, or
        its mission start if it has never published one -- AMCL stops
        emitting amcl_pose once a robot is stationary, so a robot that never
        moved has no last_point().
        """
        pt = self.dispatchers[robot_id].last_point()
        if pt is not None:
            return pt
        start = self.fleet.robots[robot_id].start
        return (start.x, start.y)

    def _check_fleet_deadlock(self) -> None:
        """Scan the currently-stuck robots for a mutual block: two stuck
        robots sitting within (a small margin over) their combined effective
        radius (robot_radius + inflation), or any robot release-gated behind
        a stuck one. Logs one consolidated FLEET DEADLOCK line per distinct
        group, then hands the group to RecoveryManager.
        """
        if (
            self.recovery is None
            or self._recovery_halted
            or self.recovery.is_recovering()
        ):
            return

        stuck_ids = sorted(rid for rid, d in self.dispatchers.items() if d.is_stuck())
        if not stuck_ids:
            return

        positions = {rid: self._robot_position(rid) for rid in stuck_ids}
        pending_blockers = {
            rid: self.release_schedule.pending_blockers(rid) for rid in self.dispatchers
        }
        group, collisions, chains = deadlock_group(
            stuck_ids,
            positions,
            self._effective_radii,
            pending_blockers,
            self._deadlock_cfg,
        )
        if not group:
            return

        if group not in self._reported_deadlocks:
            self._reported_deadlocks.add(group)
            lines = [f"coordinator_node: FLEET DEADLOCK -- robots {sorted(group)}:"]
            for a, b, gap, limit in collisions:
                lines.append(
                    f"  {a} and {b} both stuck, {gap:.2f} m apart "
                    f"(combined effective radius {limit:.2f} m)"
                )
            for rid, other, need, have in chains:
                lines.append(
                    f"  {rid} release-gated behind stuck {other} "
                    f"(needs index {need}, cleared {have})"
                )
            lines.append("  handing to coordinated recovery")
            self.get_logger().error("\n".join(lines))

        self.recovery.on_deadlock(sorted(group))


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = CoordinatorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        # _check_mission_done() can already have called rclpy.shutdown() itself
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
