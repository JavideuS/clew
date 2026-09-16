"""Per-robot dispatch: publish a robot's currently-released path to its own
local planner over ROS2, continuously, without knowing or caring what's
running underneath.

ROS2-native, straight into nav2's existing `FollowPath` action

Continuous-path model (paired with ordering.py's conflict-zone gating):
this does NOT send one waypoint at a time and wait for each to be reached.
It republishes ordering.ReleaseSchedule.released_path(robot_id) -- a
growing path *prefix* -- as a new FollowPath goal each time it's longer
than what was last sent. A new goal sent to an already-executing FollowPath
action preempts it in place, so the local planner just keeps tracking
whatever the latest path is and comes to rest at its end when nothing more
is released yet. Indistinguishable, from its point of view, from having
reached a real final goal.
That is what keeps a robot's motion smooth
except where a real conflict forces a wait.

The other direction -- turning a robot's live pose into the
`reached_index` ReleaseSchedule.report_progress() wants -- is also this
module's job: FollowPath's own feedback (distance_to_goal, speed) doesn't
carry a path index, so RobotDispatcher separately subscribes to the
robot's pose and maps it to the nearest index in that robot's own full
Spooky path.

This module is fully ROS-facing -- path_poses.py holds the pure
point/heading math it builds on (dedupe, derive headings, nearest-index),
kept ROS-free so it stays importable/testable without rclpy.

Smoothing: every released path is run through nav2's own SmoothPath action
first, and only the *smoothed* result is sent to FollowPath. Spooky's raw
output is a grid planner's path -- cell-to-cell, full of near-90-degree
corners which is normally infeasible or inefficient for robots.
Since we avoid bt_navigator, we need to smooth the path ourselves explicitly.
SmoothPath is a standalone nav2 action server (smoother_server), callable directly same
as FollowPath. If SmoothPath is rejected or times out (max_smoothing_duration), falls back to the
unsmoothed path rather than skipping the release entirely (degraded,
jerkier motion beats not moving).

Future work implies Spooky natively generating optimized smoothed paths, removing
the need for this step here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav2_msgs.action import FollowPath, SmoothPath
from nav_msgs.msg import Path
from rclpy.action import ActionClient
from rclpy.node import Node

from .geometry import yaw_to_quat_zw
from .path_poses import Pose, derive_headings, nearest_index


def _path_to_nav_msgs_path(
    points: list[tuple[float, float]],
    start_theta: float,
    goal_theta: float | None,
    frame_id: str,
    stamp,
) -> Path:
    """Thin nav_msgs/Path wrapper around path_poses.derive_headings."""
    path = Path()
    path.header.frame_id = frame_id
    path.header.stamp = stamp

    for x, y, theta in derive_headings(points, start_theta, goal_theta):
        pose = PoseStamped()
        pose.header.frame_id = frame_id
        pose.header.stamp = stamp
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.orientation.z, pose.pose.orientation.w = yaw_to_quat_zw(theta)
        path.poses.append(pose)

    return path


def _posed_path_to_nav_msgs_path(poses: list[Pose], frame_id: str, stamp) -> Path:
    """Like _path_to_nav_msgs_path but for points that already carry a
    heading -- recovery retreat / sidestep paths, whose headings are chosen
    deliberately (kept at the forward-travel tangent so a reversing
    controller backs along them), not derived from tangents here.
    """
    path = Path()
    path.header.frame_id = frame_id
    path.header.stamp = stamp
    for x, y, theta in poses:
        pose = PoseStamped()
        pose.header.frame_id = frame_id
        pose.header.stamp = stamp
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.orientation.z, pose.pose.orientation.w = yaw_to_quat_zw(theta)
        path.poses.append(pose)
    return path


@dataclass(frozen=True)
class DispatchConfig:
    """Deployment wiring + tuning for RobotDispatcher, shared across every
    robot in the fleet (per-robot values -- id, path, headings -- stay
    separate constructor args, they're not the same fact for every robot
    the way these are). `{robot_id}` in the templates is filled in
    per-robot by RobotDispatcher itself.
    """

    frame_id: str = "map"
    action_name_template: str = "/{robot_id}/follow_path"
    smooth_action_name_template: str = "/{robot_id}/smooth_path"
    pose_topic_template: str = "/{robot_id}/amcl_pose"
    max_smoothing_duration_s: float = 2.0
    # After this many controller aborts in a row on the same released
    # prefix, stop re-sending it (see _on_follow_path_result) and fire
    # on_stuck once, instead of looping forever.
    max_consecutive_aborts: int = 5


class RobotDispatcher:
    """One instance per robot: owns its FollowPath action client, its pose
    subscription, and the length of the last path it actually sent.
    """

    def __init__(
        self,
        robot_id: str,
        node: Node,
        full_path: list[tuple[float, float]],
        start_theta: float,
        goal_theta: float,
        on_progress: Callable[[int], None],
        cfg: DispatchConfig = DispatchConfig(),
        on_stuck: Callable[[str, int], None] | None = None,
    ) -> None:
        """Args:
        robot_id: this robot's id, used to fill the action/topic templates.
        node: the coordinator's rclpy node (owns the action client's/
            subscription's callbacks -- this class doesn't spin its own).
        full_path: this robot's complete Spooky path (not just what's
            released yet) -- needed to map a live pose to a path index
            regardless of how much has been released so far.
        start_theta, goal_theta: the robot's real start/goal heading (from
            Fleet's Robot.start.theta/goal.theta). start_theta is always
            pinned onto the first pose of every release; goal_theta is only
            pinned onto the last pose when the release is the full path
        on_progress: called with the nearest path index every time a new
            pose arrives. Wire this to
            `lambda idx: release_schedule.report_progress(robot_id, idx)`
            in coordinator_node.py -- this class only computes the index,
            it doesn't own the ReleaseSchedule.
        cfg: action/topic name templates and smoothing/retry tuning, the
            same across every robot -- see DispatchConfig.
        on_stuck: called (robot_id, abort_count) the first time
            cfg.max_consecutive_aborts is hit for the current stuck
            prefix. Wire it to a coordinator-level diagnostic / replan
            trigger; the dispatcher itself just stops hammering and waits
            for more path to be released.
        """
        self._robot_id = robot_id
        self._node = node
        self._full_path = full_path
        self._start_theta = start_theta
        self._goal_theta = goal_theta
        self._frame_id = cfg.frame_id
        self._on_progress = on_progress
        self._max_smoothing_duration_s = cfg.max_smoothing_duration_s
        self._max_consecutive_aborts = cfg.max_consecutive_aborts
        self._on_stuck = on_stuck
        self._stuck = False
        # Last pose seen for this robot (map frame) -- the coordinator
        # reads it to spot mutual deadlocks.
        self._last_point: tuple[float, float] | None = None
        # Set by an under-cap abort to force one re-send of the current
        # prefix on the next tick, without rewinding _sent_length (which
        # would lose track of how far we've genuinely progressed and let
        # every retry reset the abort counter).
        self._resend_pending = False
        self._sent_length = 0
        # Set alongside _sent_length each publish_released_path -- whether
        # *that* send covered the whole path, not just a released prefix.
        # _on_follow_path_result checks it on SUCCEEDED to know whether this
        # robot has actually finished its mission, not just one segment of
        # it (coordinator_node polls is_complete() fleet-wide for that).
        self._sent_is_final = False
        self._completed = False
        self._goal_handle = None
        # Recovery hooks. While paused, publish_released_path is a no-op (the
        # release schedule keeps advancing but nothing is sent). A one-off
        # path (recovery retreat / sidestep) is tracked by its own goal seq
        # so its terminal result routes to the recovery callback, not the
        # normal retry logic.
        self._paused = False
        self._oneoff_seq = -1
        self._oneoff_done_cb: Callable[[bool], None] | None = None
        # Bumped on every FollowPath send. A result callback ignores any
        # result whose seq isn't the current one, so a preempted goal's
        # CANCELED/ABORTED (a newer goal replaced it in place) isn't
        # mistaken for a real failure of the goal that's actually running.
        self._goal_seq = 0
        self._active_goal_seq = 0
        self._consecutive_aborts = 0

        action_name = cfg.action_name_template.format(robot_id=robot_id)
        self._action_client = ActionClient(node, FollowPath, action_name)

        smooth_action_name = cfg.smooth_action_name_template.format(robot_id=robot_id)
        self._smoother_client = ActionClient(node, SmoothPath, smooth_action_name)

        pose_topic = cfg.pose_topic_template.format(robot_id=robot_id)
        self._pose_sub = node.create_subscription(
            PoseWithCovarianceStamped, pose_topic, self._on_pose, 10
        )

    def publish_released_path(self, released_path: list[tuple[float, float]]) -> None:
        """Smooth `released_path` via nav2's SmoothPath, then send the
        result as a new FollowPath goal -- but only if it's grown since the
        last publish; a no-op otherwise, so callers can call this on every
        tick without spamming redundant goals/smoothing requests.
        """
        if self._paused:
            return

        grew = len(released_path) > self._sent_length
        if not grew and not self._resend_pending:
            return

        if grew:
            # A genuinely longer prefix has been released -- real progress,
            # so drop any abort state. If we'd given up on a stuck prefix
            # (max_consecutive_aborts hit, _sent_length left untouched so
            # this method kept quiet), the gate it was waiting on has since
            # opened: resume.
            if self._stuck:
                self._node.get_logger().info(
                    f"dispatch: {self._robot_id} resuming -- "
                    f"{len(released_path)} path point(s) now released, past "
                    "the prefix that was stuck; retrying"
                )
            self._stuck = False
            self._consecutive_aborts = 0

        if not self._smoother_client.server_is_ready():
            self._node.get_logger().warning(
                f"dispatch: {self._robot_id}'s SmoothPath action server "
                "not available yet -- skipping this publish, will retry "
                "on the next tick"
            )
            return

        if not self._action_client.server_is_ready():
            self._node.get_logger().warning(
                f"dispatch: {self._robot_id}'s FollowPath action server "
                "not available yet -- skipping this publish, will retry "
                "on the next tick"
            )
            return

        # Only the genuinely final release gets the real mission goal
        # heading pinned onto its last point -- see path_poses.derive_headings'
        # own docstring for the bug this guards against (every partial
        # release's endpoint looking like a real arrival otherwise).
        is_final_release = len(released_path) >= len(self._full_path)
        effective_goal_theta = self._goal_theta if is_final_release else None

        raw_path_msg = _path_to_nav_msgs_path(
            released_path,
            self._start_theta,
            effective_goal_theta,
            self._frame_id,
            self._node.get_clock().now().to_msg(),
        )

        # Marked "handled" here, at request-issue-time, not once FollowPath
        # actually gets sent below -- otherwise a tick landing mid-round-trip
        # (smoothing is two async hops away from FollowPath) would see this
        # length as still un-sent and fire a redundant, overlapping
        # SmoothPath request for the same path. _sent_length only ever grows
        # (an under-cap abort retries via _resend_pending, not by rewinding
        # it), so an abort counter survives across retries of one prefix.
        self._sent_length = max(self._sent_length, len(released_path))
        self._sent_is_final = is_final_release
        self._resend_pending = False

        self._smooth_then_follow(raw_path_msg, oneoff=False)

    def _smooth_then_follow(self, raw_path_msg, *, oneoff: bool) -> None:
        """Request nav2's SmoothPath on `raw_path_msg`, then send whatever
        comes back (or `raw_path_msg` itself, on any smoothing failure) as
        a FollowPath goal. Shared by every path this dispatcher can send --
        the release-schedule publish path and recovery's one-off moves
        alike.
        Nothing should reach FollowPath unsmoothed.
        """
        smooth_goal = SmoothPath.Goal()
        smooth_goal.path = raw_path_msg
        duration_s = self._max_smoothing_duration_s
        smooth_goal.max_smoothing_duration.sec = int(duration_s)
        smooth_goal.max_smoothing_duration.nanosec = int((duration_s % 1.0) * 1e9)
        # A region-dependent "computed but rejected" failure (smoother_server
        # internally produces and even publishes a valid plan_smoothed, but
        # the actual action result comes back empty) is exactly the shape
        # collision-check rejection during smoothing would produce, and
        # local_costmap-based avoidance already runs during actual FollowPath
        # execution regardless -- smoothing doesn't need to duplicate it.
        smooth_goal.check_for_collisions = False

        self._node.get_logger().info(
            f"dispatch: {self._robot_id} sending SmoothPath request with "
            f"{len(raw_path_msg.poses)} input pose(s)"
            f"{' (one-off)' if oneoff else ''}"
        )
        for i, pose in enumerate(raw_path_msg.poses):
            self._node.get_logger().debug(
                f"dispatch: {self._robot_id} pose {i}: "
                f"{pose.pose.position.x}, {pose.pose.position.y}"
            )

        send_future = self._smoother_client.send_goal_async(smooth_goal)
        send_future.add_done_callback(
            lambda future: self._on_smooth_goal_response(
                future, raw_path_msg, oneoff=oneoff
            )
        )

    def _on_smooth_goal_response(
        self, future, fallback_path_msg, *, oneoff: bool = False
    ) -> None:
        goal_handle = future.result()
        if not goal_handle.accepted:
            self._node.get_logger().warning(
                f"dispatch: {self._robot_id}'s SmoothPath goal was "
                "rejected -- falling back to the unsmoothed path"
            )
            self._send_follow_path_goal(fallback_path_msg, oneoff=oneoff)
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda result_future: self._on_smooth_result(
                result_future, fallback_path_msg, oneoff=oneoff
            )
        )

    def _on_smooth_result(
        self, result_future, fallback_path_msg, *, oneoff: bool = False
    ) -> None:
        wrapped = result_future.result()
        result = wrapped.result
        self._node.get_logger().info(
            f"dispatch: {self._robot_id}'s SmoothPath returned: "
            f"action_status={wrapped.status}, was_completed={result.was_completed}, "
            f"output_pose_count={len(result.path.poses)}"
        )
        if not result.was_completed:
            self._node.get_logger().warning(
                f"dispatch: {self._robot_id}'s SmoothPath did not complete "
                "within max_smoothing_duration. Falling back to the "
                "unsmoothed path"
            )
            path_msg = fallback_path_msg
        elif not result.path.poses:
            # was_completed only means the smoother finished within its
            # time budget, not that what it produced is usable.
            self._node.get_logger().warning(
                f"dispatch: {self._robot_id}'s SmoothPath completed but "
                "returned an empty path. Falling back to the unsmoothed "
                "path"
            )
            path_msg = fallback_path_msg
        else:
            path_msg = result.path
        self._send_follow_path_goal(path_msg, oneoff=oneoff)

    def _send_follow_path_goal(self, path_msg, *, oneoff: bool = False) -> None:
        goal = FollowPath.Goal()
        goal.path = path_msg

        self._goal_seq += 1
        seq = self._goal_seq
        self._active_goal_seq = seq
        if oneoff:
            self._oneoff_seq = seq

        self._node.get_logger().info(
            f"dispatch: {self._robot_id} sending "
            f"{'one-off ' if oneoff else ''}FollowPath goal #{seq} with "
            f"{len(path_msg.poses)} pose(s)"
        )

        # No explicit cancel first -- a new goal preempts the running one.
        send_future = self._action_client.send_goal_async(goal)
        send_future.add_done_callback(
            lambda future: self._on_goal_response(future, seq)
        )

    def _on_goal_response(self, future, seq: int) -> None:
        goal_handle = future.result()
        if not goal_handle.accepted:
            self._node.get_logger().error(
                f"dispatch: {self._robot_id}'s FollowPath goal #{seq} was rejected"
            )
            return
        self._goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda result_future: self._on_follow_path_result(result_future, seq)
        )

    def _on_follow_path_result(self, future, seq: int) -> None:
        """Self-healing retry, capped. A FollowPath goal that's accepted and
        then aborted ("Resulting plan has 0 poses in it", or the controller
        giving up on a blocked path) strands the robot forever: _sent_length
        was advanced at request-issue time and released_path() won't exceed
        it again unless *more* path is released, which for a robot that
        never moved never happens. Setting _resend_pending makes the next
        _dispatch_tick re-send the current released prefix (without
        rewinding _sent_length, so the abort count below survives the
        retry); the tick period paces the retry until localization catches
        up and the goal sticks.

        That retry is bounded by max_consecutive_aborts. A transient cause
        (localization still settling, a one-off empty smoother result)
        clears in a retry or two; past the cap it's structural -- most
        often this robot's gated hold pose is too close to another stopped
        robot for either controller to move (see ordering.py's history
        note). Re-sending the identical failing prefix hundreds of times
        just buries the logs, so instead: stop, leave _sent_length alone so
        publish_released_path stays quiet until the schedule releases *more*
        path (whatever this robot was gated behind having finally cleared),
        and fire on_stuck once so the coordinator can report why / replan.

        Results from superseded goals (seq != active) are ignored:
        preempting a running FollowPath surfaces the old goal's result as
        CANCELED/ABORTED, which isn't a failure. A CANCELED result on the
        *current* goal is our own cancel() call -- also not a retry.
        """
        status = future.result().status

        if seq == self._oneoff_seq:
            # A recovery retreat / sidestep goal -- its result belongs to
            # the recovery callback, not the normal gated-dispatch retry.
            self._oneoff_seq = -1
            cb, self._oneoff_done_cb = self._oneoff_done_cb, None
            succeeded = status == GoalStatus.STATUS_SUCCEEDED
            self._node.get_logger().info(
                f"dispatch: {self._robot_id}'s one-off goal #{seq} ended "
                f"({'succeeded' if succeeded else f'status {status}'})"
            )
            if cb is not None:
                cb(succeeded)
            return

        if seq != self._active_goal_seq:
            return

        if status == GoalStatus.STATUS_SUCCEEDED:
            self._consecutive_aborts = 0
            self._stuck = False
            if self._sent_is_final:
                self._completed = True
            self._node.get_logger().info(
                f"dispatch: {self._robot_id}'s FollowPath goal #{seq} "
                "succeeded (reached the end of the released prefix)"
            )
            return

        if status == GoalStatus.STATUS_ABORTED:
            self._consecutive_aborts += 1

            if self._consecutive_aborts < self._max_consecutive_aborts:
                self._node.get_logger().warning(
                    f"dispatch: {self._robot_id}'s FollowPath goal #{seq} was "
                    f"aborted by the controller (attempt "
                    f"{self._consecutive_aborts}/{self._max_consecutive_aborts}) "
                    "-- will re-send the current released prefix on the next "
                    "tick"
                )
                self._resend_pending = True
                return

            if not self._stuck:
                self._node.get_logger().error(
                    f"dispatch: {self._robot_id}'s FollowPath aborted "
                    f"{self._consecutive_aborts} times in a row on the same "
                    "released prefix -- giving up on re-sending it. Holding "
                    "until more path is released or a replan occurs."
                )
                self._stuck = True
                if self._on_stuck is not None:
                    self._on_stuck(self._robot_id, self._consecutive_aborts)
            return

        self._node.get_logger().info(
            f"dispatch: {self._robot_id}'s FollowPath goal #{seq} ended with "
            f"status {status}; not retrying"
        )

    def _on_pose(self, msg: PoseWithCovarianceStamped) -> None:
        point = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        self._last_point = point
        # While stuck (abort cap hit, no goal running), don't keep feeding
        # progress into the ReleaseSchedule: a stuck robot must not go on
        # satisfying other robots' gates -- it may be blocking them, and
        # recovery may need to walk it back. Progress resumes once a longer
        # prefix is released and publish_released_path clears _stuck.
        if self._stuck:
            return
        self._on_progress(nearest_index(point, self._full_path))

    def is_stuck(self) -> bool:
        """True once this robot has hit the consecutive-abort cap and given
        up re-sending its current prefix (cleared when more path is
        released). The coordinator polls this for deadlock detection."""
        return self._stuck

    def last_point(self) -> tuple[float, float] | None:
        """Most recent (x, y) pose seen for this robot, or None if none yet."""
        return self._last_point

    def full_path(self) -> list[tuple[float, float]]:
        """This robot's complete current path (start .. goal)."""
        return list(self._full_path)

    def is_complete(self) -> bool:
        """True once a FollowPath goal covering this robot's *entire*
        current path (not just a released prefix) has succeeded. Cleared
        by update_path -- a recovery replan means there's a new path to
        finish, so a stale True here would let coordinator_node think the
        mission was done when a robot had actually just been re-routed."""
        return self._completed

    def cancel(self) -> None:
        """Cancel this robot's outstanding goal, if any."""
        if self._goal_handle is not None:
            self._goal_handle.cancel_goal_async()

    # ── recovery hooks (driven by coordinator_node.RecoveryManager) ─────────

    def trail(self) -> list[tuple[float, float]]:
        """The prefix of this robot's full path it has actually driven,
        start .. current position (nearest full-path point to its last
        pose). Used by recovery to find a clear point to back up to.
        """
        ref = self._last_point or self._full_path[0]
        idx = nearest_index(ref, self._full_path)
        return list(self._full_path[: idx + 1])

    def remaining(self) -> list[tuple[float, float]]:
        """This robot's full path from its current position onward -- the
        route recovery must keep a yielder clear of.
        """
        ref = self._last_point or self._full_path[0]
        idx = nearest_index(ref, self._full_path)
        return list(self._full_path[idx:])

    def pause(self) -> None:
        """Stop gated dispatch and cancel any running gated goal. A one-off
        recovery path can still be sent while paused (it's the explicit
        override). Cleared by update_path() or resume().
        """
        self._paused = True
        self.cancel()

    def resume(self) -> None:
        self._paused = False

    def update_path(self, new_full_path: list[tuple[float, float]]) -> None:
        """Swap in a freshly planned path (recovery replan) and clear all
        transient dispatch state so the next tick re-sends from scratch
        against the new ReleaseSchedule.
        """
        self._full_path = new_full_path
        self._sent_length = 0
        self._sent_is_final = False
        self._completed = False
        self._resend_pending = False
        self._stuck = False
        self._consecutive_aborts = 0
        self._paused = False

    def send_oneoff_path(
        self,
        poses: list[Pose],
        on_done: Callable[[bool], None],
    ) -> None:
        """Send `poses` (already carrying headings) as a single FollowPath
        goal, bypassing the release schedule -- but still routed through
        SmoothPath like every other goal, since these are recovery moves
        over the same raw Spooky grid paths that need smoothing to not
        stall the controller. `on_done` is called with True/False when the
        goal terminates. Used by recovery for retreat / sidestep moves.
        """
        if len(poses) < 2:
            on_done(True)  # nothing to do -- already where it needs to be
            return
        if not self._smoother_client.server_is_ready():
            self._node.get_logger().warning(
                f"dispatch: {self._robot_id}'s SmoothPath server not ready "
                "for a one-off recovery move"
            )
            on_done(False)
            return
        if not self._action_client.server_is_ready():
            self._node.get_logger().warning(
                f"dispatch: {self._robot_id}'s FollowPath server not ready "
                "for a one-off recovery move"
            )
            on_done(False)
            return
        self._oneoff_done_cb = on_done
        raw_path_msg = _posed_path_to_nav_msgs_path(
            poses, self._frame_id, self._node.get_clock().now().to_msg()
        )
        self._smooth_then_follow(raw_path_msg, oneoff=True)
