"""ConvergenceGate: one-shot AMCL bootstrap for a fleet.
Seed every
robot's initial pose, poll TF until each one's localization has settled
on its seeded start (or a timeout elapses), then hand off via a single
on_converged() callback and tear down (clear) its own timer, TF subscriptions,
and seed publishers.

This class owns only the
ROS-facing plumbing: TF buffers, publishers, timers.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable

from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.clock import Clock, ClockType
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile
from rclpy.time import Time
from tf2_msgs.msg import TFMessage
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer

from .geometry import angle_diff, distance, quat_to_yaw, yaw_to_quat_zw
from .monitoring import (
    ConvergenceConfig,
    convergence_offset,
    convergence_streak_update,
    systematic_offset,
)
from .robot import Fleet

# tf_static is latched (transient-local); a subscriber must match that
# durability or it misses transforms published before it subscribed.
_TF_STATIC_QOS = QoSProfile(
    depth=100,
    history=QoSHistoryPolicy.KEEP_LAST,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
)


def _initial_pose_covariance() -> list[float]:
    """RViz's own "2D Pose Estimate" default covariance (6x6 diagonal,
    only x/y/yaw set) -- reused so a seeded pose looks the same to AMCL
    as a manual click would. A fresh list every call, not shared mutable
    module state.
    """
    cov = [0.0] * 36
    cov[0] = 0.25  # x
    cov[7] = 0.25  # y
    cov[35] = 0.06853892326654787  # yaw
    return cov


class ConvergenceGate:
    """Seed each robot's AMCL with its mission-declared start, then gate on every
    robot's localization actually converging on that seed.

    The mission's declared start is only a trustworthy seed when it's the
    same fact as the robot's actual position by construction. Not recommended to
    use on real hardware if you are not certain about initial pose.

    Call start() once; on_converged() fires exactly once, either because
    every robot converged or because cfg.timeout_s elapsed -- at which
    point this gate destroys its own timer, TF subscriptions, and seed
    publishers (nothing here is needed again for the rest of the node's
    life) before calling on_converged().

    "Converged" means map->base_frame has stopped moving between polls
    (cfg.settle_pos_tol_m/settle_yaw_tol_rad), NOT that it landed on the
    declared seed. A wrong-but-plausible declared start is a kick-start
    for AMCL's particle filter, not a promise -- real sensor data will
    correctly scan-match away from it if it's off, the same as it would
    for a slightly-off RViz "2D Pose Estimate" click, and gating on
    "matches the declared value" would just time out forever in that
    case.

    Once settled, if that's still farther than pos_tol_m/yaw_tol_rad from
    the declared start, this re-seeds at an extrapolated point (assuming
    AMCL's pull here is roughly constant nearby) and tries again, up to
    cfg.max_correction_rounds times, keeping whichever round landed
    closest. It is an attempt to compensate for AMCL's drift back toward the
    operator's actually-known position, not a guarantee (a target that's
    genuinely implausible to the map/sensors won't converge no matter how
    many rounds are spent chasing it).

    settled_poses exposes each robot's resulting map->base_frame pose,
    for the caller to plan from instead of the possibly-stale mission-
    file start outright.
    """

    def __init__(
        self,
        node: Node,
        fleet: Fleet,
        cfg: ConvergenceConfig,
        on_converged: Callable[[], None],
        *,
        topic_template: str = "/{robot_id}/initialpose",
        tf_topic_template: str = "/{robot_id}/tf",
        tf_static_topic_template: str = "/{robot_id}/tf_static",
        map_frame: str = "map",
        base_frame_template: str = "base_link",
    ) -> None:
        self._node = node
        self._fleet = fleet
        self._cfg = cfg
        self._on_converged = on_converged
        self._topic_template = topic_template
        self._tf_topic_template = tf_topic_template
        self._tf_static_topic_template = tf_static_topic_template
        self._map_frame = map_frame
        self._base_frame_template = base_frame_template

        # Wall-clock, not node.get_clock(), a real wait on AMCL's physical-time compute, unrelated
        # to how fast the sim clock runs.
        self._wall_clock = Clock(clock_type=ClockType.SYSTEM_TIME)

        self._seed_publishers = {}
        self._seed_msgs = {}
        self._tf_buffers: dict[str, Buffer] = {}
        self._tf_subs = []  # keep subscriptions alive past __init__
        self._base_frames = {}
        self._converged: set[str] = set()
        self._convergence_streak = {robot.id: 0 for robot in fleet}
        # Settle-detection state: last poll's (x, y, yaw) per robot (the
        # reference a new sample is compared against for "did it move")
        # and each robot's pose at the moment it stopped moving, once
        # converged -> see settled_poses.
        self._last_stable_ref: dict[str, tuple[float, float, float]] = {}
        self._settled_poses: dict[str, tuple[float, float, float]] = {}
        # Offset-compensation state (see _poll's converged branch):
        # correction rounds used so far per robot, and the closest-to-
        # declared settle observed across all of them, kept in case
        # max_correction_rounds is exhausted before landing in tolerance.
        self._correction_round: dict[str, int] = {robot.id: 0 for robot in fleet}
        self._best_settled: dict[str, tuple[float, float, float]] = {}
        self._best_settled_dist: dict[str, float] = {}
        self._poll_count = 0
        self._done = False
        # Recent (TF pose - seeded start) samples per robot, for the
        # timeout diagnostic in _report_offset_diagnostics.
        self._offset_history: dict[str, deque] = {
            robot.id: deque(maxlen=cfg.offset_history_len) for robot in fleet
        }
        self._timer = None
        self._deadline = None

    def start(self) -> None:
        for robot in self._fleet:
            buf = Buffer()
            self._tf_buffers[robot.id] = buf
            self._base_frames[robot.id] = self._base_frame_template.format(
                robot_id=robot.id
            )

            # Manual TF ingestion rather than a TransformListener: the
            # listener subscribes to tf/tf_static relative to this
            # (un-namespaced) node, but each robot's tree is on its own
            # /{robot_id}/tf topic. Default-arg capture of `buf` for the
            # same per-iteration-binding reason as CoordinatorNode's own
            # on_progress lambda in _plan_fleet.
            self._tf_subs.append(
                self._node.create_subscription(
                    TFMessage,
                    self._tf_topic_template.format(robot_id=robot.id),
                    lambda msg, b=buf: self._ingest_tf(msg, b, static=False),
                    10,
                )
            )
            self._tf_subs.append(
                self._node.create_subscription(
                    TFMessage,
                    self._tf_static_topic_template.format(robot_id=robot.id),
                    lambda msg, b=buf: self._ingest_tf(msg, b, static=True),
                    _TF_STATIC_QOS,
                )
            )

            topic = self._topic_template.format(robot_id=robot.id)
            pub = self._node.create_publisher(PoseWithCovarianceStamped, topic, 1)
            msg = PoseWithCovarianceStamped()
            msg.header.frame_id = "map"
            msg.pose.pose.position.x = robot.start.x
            msg.pose.pose.position.y = robot.start.y
            msg.pose.pose.orientation.z, msg.pose.pose.orientation.w = yaw_to_quat_zw(
                robot.start.theta
            )
            msg.pose.covariance = _initial_pose_covariance()
            self._seed_publishers[robot.id] = pub
            self._seed_msgs[robot.id] = msg
            self._publish_seed(robot.id)
            self._node.get_logger().info(
                f"coordinator_node: seeded initial pose for {robot.id} on "
                f"{topic!r}: ({robot.start.x}, {robot.start.y}) -- waiting for "
                "its localization to converge there (checked in TF)"
            )

        self._deadline = self._wall_clock.now() + Duration(seconds=self._cfg.timeout_s)
        self._timer = self._node.create_timer(
            self._cfg.poll_period_s, self._poll, clock=self._wall_clock
        )

    def _publish_seed(self, robot_id: str) -> None:
        msg = self._seed_msgs[robot_id]
        msg.header.stamp = self._node.get_clock().now().to_msg()
        self._seed_publishers[robot_id].publish(msg)

    def _reseed_at(self, robot_id: str, x: float, y: float, yaw: float) -> None:
        """Overwrite this robot's seed message's pose in place (the
        covariance and everything else about it stays what start() set
        up) and publish it -- used for offset-compensation rounds, where
        the seed itself needs to move, not just be re-sent unchanged.
        """
        msg = self._seed_msgs[robot_id]
        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        msg.pose.pose.orientation.z, msg.pose.pose.orientation.w = yaw_to_quat_zw(yaw)
        self._publish_seed(robot_id)

    def _ingest_tf(self, msg: TFMessage, buf: Buffer, *, static: bool) -> None:
        setter = buf.set_transform_static if static else buf.set_transform
        for transform in msg.transforms:
            setter(transform, "clew")

    def _poll(self) -> None:
        if self._done:
            return  # a queued callback can still fire once after _finish()

        cfg = self._cfg
        self._poll_count += 1
        count = self._poll_count
        reseed = (
            count <= cfg.reseed_burst_polls or count % cfg.reseed_every_n_polls == 0
        )

        for robot in self._fleet:
            if robot.id in self._converged:
                continue  # never re-seed a converged robot -- resets its covariance

            if reseed:
                self._publish_seed(robot.id)

            blocker = self._blocker(robot)
            streak, converged = convergence_streak_update(
                self._convergence_streak[robot.id], blocker is not None, cfg
            )
            self._convergence_streak[robot.id] = streak
            if blocker is not None and count % cfg.log_every_n_polls == 0:
                self._node.get_logger().warning(
                    f"coordinator_node: {robot.id} not localized yet -- {blocker}"
                )

            if converged:
                self._on_settled(robot, cfg)

        if all(robot.id in self._converged for robot in self._fleet):
            self._node.get_logger().info(
                "coordinator_node: whole fleet localized at its declared "
                "starts -- proceeding to plan"
            )
            self._finish()
            return

        if self._wall_clock.now() >= self._deadline:
            unconverged = sorted(
                r.id for r in self._fleet if r.id not in self._converged
            )
            self._node.get_logger().error(
                "coordinator_node: localization did NOT converge for "
                f"{unconverged} within {cfg.timeout_s}s -- planning "
                "anyway. Cross-robot release timing for those robots may be "
                "off until their localization settles; make this a hard abort "
                "instead if a corrupted timeline is worse than a late start."
            )
            self._report_offset_diagnostics(unconverged)
            self._finish()

    def _blocker(self, robot) -> str | None:
        """None once this robot's map -> base_frame transform exists and
        has stopped moving between polls (within cfg.settle_pos_tol_m /
        cfg.settle_yaw_tol_rad of the previous poll's reading).
        Otherwise a short string naming what's still wrong, for the
        throttled diagnostic log in _poll. Caller requires None to hold
        cfg.stable_polls polls in a row before trusting it.

        No transform-age check: AMCL republishes map->odom continuously
        (future-dated by transform_tolerance), so a stale *value* still
        carries a fresh stamp -- and comparing the stamp against this
        node's clock is a footgun when sim time and wall time disagree.
        The movement check below is the real gate; a genuinely absent
        tree surfaces as the lookup raising instead.
        """
        buf = self._tf_buffers[robot.id]
        base = self._base_frames[robot.id]
        try:
            tf = buf.lookup_transform(self._map_frame, base, Time())
        except TransformException as exc:
            known = " ".join(buf.all_frames_as_string().split()) or "<none>"
            return (
                f"no {self._map_frame} -> {base} transform yet ({exc!s}); "
                f"frames in buffer: {known}"
            )

        t = tf.transform.translation
        q = tf.transform.rotation
        yaw = quat_to_yaw(q.x, q.y, q.z, q.w)

        # Declared-vs-actual offset: diagnostics only now (the timeout
        # report's systematic-frame-error check), not the gate itself.
        # Signed samples (not abs) -- that report needs the mean
        # direction to tell a systematic offset from scatter.
        dx, dy, dyaw, _ = convergence_offset(
            (t.x, t.y),
            yaw,
            (robot.start.x, robot.start.y, robot.start.theta),
            self._cfg,
        )
        self._offset_history[robot.id].append((dx, dy, dyaw))

        prev = self._last_stable_ref.get(robot.id)
        self._last_stable_ref[robot.id] = (t.x, t.y, yaw)
        if prev is None:
            return "waiting for a second reading to check whether it's settled"

        step_dist = distance((t.x, t.y), (prev[0], prev[1]))
        step_dyaw = angle_diff(yaw, prev[2])
        if (
            step_dist > self._cfg.settle_pos_tol_m
            or abs(step_dyaw) > self._cfg.settle_yaw_tol_rad
        ):
            return (
                f"still settling -- moved {step_dist:.2f} m / "
                f"{abs(step_dyaw):.2f} rad since the last poll "
                f"(tol {self._cfg.settle_pos_tol_m} m / "
                f"{self._cfg.settle_yaw_tol_rad} rad)"
            )
        return None

    def _on_settled(self, robot, cfg: ConvergenceConfig) -> None:
        """Called once this robot's map->base_frame has just reached
        cfg.stable_polls polls without moving (see _blocker). Either
        finalizes it as converged (declared_settled_poses), or -- if
        still farther than pos_tol_m/yaw_tol_rad from the declared start
        and correction rounds remain -- re-seeds at an extrapolated
        point and keeps polling; see ConvergenceConfig.max_correction_rounds.
        """
        settled = self._last_stable_ref[robot.id]
        target = (robot.start.x, robot.start.y, robot.start.theta)
        off_dist = distance(settled[:2], target[:2])
        off_yaw = abs(angle_diff(settled[2], target[2]))

        if off_dist < self._best_settled_dist.get(robot.id, math.inf):
            self._best_settled_dist[robot.id] = off_dist
            self._best_settled[robot.id] = settled

        close_enough = off_dist <= cfg.pos_tol_m and off_yaw <= cfg.yaw_tol_rad
        round_ = self._correction_round[robot.id]
        if close_enough or round_ >= cfg.max_correction_rounds:
            self._converged.add(robot.id)
            self._settled_poses[robot.id] = self._best_settled[robot.id]
            bx, by, _ = self._settled_poses[robot.id]
            if close_enough:
                self._node.get_logger().info(
                    f"coordinator_node: {robot.id}'s localization settled "
                    f"at ({bx:.2f}, {by:.2f}), within tolerance of its "
                    "declared start"
                )
            else:
                self._node.get_logger().warning(
                    f"coordinator_node: {robot.id}'s localization never "
                    f"settled within tolerance of its declared start after "
                    f"{round_} correction round(s) -- using its closest "
                    f"settle ({bx:.2f}, {by:.2f}), "
                    f"{self._best_settled_dist[robot.id]:.2f} m off"
                )
            return

        self._correction_round[robot.id] = round_ + 1
        comp_x = target[0] - (settled[0] - target[0])
        comp_y = target[1] - (settled[1] - target[1])
        comp_yaw = target[2] - angle_diff(settled[2], target[2])
        self._node.get_logger().info(
            f"coordinator_node: {robot.id} settled {off_dist:.2f} m from "
            f"its declared start -- re-seeding at ({comp_x:.2f}, "
            f"{comp_y:.2f}) to compensate (round {round_ + 1}/"
            f"{cfg.max_correction_rounds})"
        )
        self._reseed_at(robot.id, comp_x, comp_y, comp_yaw)
        # Fresh settle-detection state: the next poll's _blocker must judge
        # movement against the new seed's reaction, not the old one's.
        self._convergence_streak[robot.id] = 0
        self._last_stable_ref.pop(robot.id, None)

    @property
    def settled_poses(self) -> dict[str, tuple[float, float, float]]:
        """(x, y, yaw) per robot that reached settled (map->base_frame
        stopped moving) and either landed within cfg.pos_tol_m/
        cfg.yaw_tol_rad of its declared start, or exhausted
        cfg.max_correction_rounds trying to -- the closest one of those
        rounds wins either way. The caller's best available estimate of
        where that robot actually is, to plan from instead of the
        declared mission-file start outright. Robots that never settled
        at all (timed out) are absent; the caller should keep using their
        declared start for those.
        """
        return dict(self._settled_poses)

    def _report_offset_diagnostics(self, unconverged: list[str]) -> None:
        """On convergence timeout, summarise each stuck robot's mean
        (TF - seed) offset and whether the offsets look systematic
        (consistent across the fleet -> map origin / sensor mount / frame
        bug) or scattered (genuine localization difficulty -- sensor
        noise, poor map, ambiguous geometry).
        """
        means: dict[str, tuple[float, float]] = {}
        for rid in unconverged:
            hist = self._offset_history.get(rid)
            if not hist:
                self._node.get_logger().error(
                    f"coordinator_node:   {rid}: no TF pose ever received -- "
                    "check the tf topic/frame params, not localization"
                )
                continue
            n = len(hist)
            mx = sum(s[0] for s in hist) / n
            my = sum(s[1] for s in hist) / n
            myaw = sum(s[2] for s in hist) / n
            spread = math.sqrt(
                sum((s[0] - mx) ** 2 + (s[1] - my) ** 2 for s in hist) / n
            )
            means[rid] = (mx, my)
            self._node.get_logger().error(
                f"coordinator_node:   {rid}: mean offset ({mx:+.2f}, {my:+.2f}) m, "
                f"{myaw:+.2f} rad; jitter {spread:.2f} m over {n} samples"
            )

        if systematic_offset(means, self._cfg):
            self._node.get_logger().error(
                "coordinator_node:   -> same offset on every robot: this is a "
                "systematic frame error (map origin, or a sensor prim offset "
                "from the frame it publishes as), NOT localization noise"
            )
        elif len(means) >= 2:
            self._node.get_logger().error(
                "coordinator_node:   -> offsets differ per robot: genuine "
                "localization difficulty (sensor noise, map quality, "
                "ambiguous geometry)"
            )

    def _finish(self) -> None:
        """Clear all resources and callback on convergence.
        They won't be needed anymore
        """
        self._done = True
        if self._timer is not None:
            self._timer.cancel()
            self._node.destroy_timer(self._timer)
        for sub in self._tf_subs:
            self._node.destroy_subscription(sub)
        for pub in self._seed_publishers.values():
            self._node.destroy_publisher(pub)
        self._tf_subs = []
        self._seed_publishers = {}
        self._tf_buffers = {}
        self._on_converged()
