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

from .geometry import quat_to_yaw, yaw_to_quat_zw
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
    """Seed each robot's AMCL with its mission-declared start (standing
    in for RViz's manual "2D Pose Estimate" click), then gate on every
    robot's localization actually converging on that seed -- see
    ConvergenceConfig for what "converged" means and why it's checked in
    TF rather than via amcl_pose.

    The mission's declared start is only a trustworthy seed when it's the
    same fact as the robot's actual position by construction -- true in
    sim (you control both), NOT true on a real robot, where the declared
    start is an assumption, not a measurement ("Planning never reads the
    robot's actual pose" finding -- same category of bug, different
    symptom: there it broke MPC's trajectory reference; here, left
    unpublished, it breaks nav2's local-costmap path pruning instead).
    Don't use this against real hardware unless the declared start is
    independently guaranteed accurate (e.g. robots placed at known
    docks) -- otherwise wire a real pose source (an operator
    confirmation, a UI click on the map, ...) in before start() instead
    of trusting the mission file.

    Call start() once; on_converged() fires exactly once, either because
    every robot converged or because cfg.timeout_s elapsed -- at which
    point this gate destroys its own timer, TF subscriptions, and seed
    publishers (nothing here is needed again for the rest of the node's
    life) before calling on_converged().
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

    def _ingest_tf(self, msg: TFMessage, buf: Buffer, *, static: bool) -> None:
        setter = buf.set_transform_static if static else buf.set_transform
        for transform in msg.transforms:
            setter(transform, "fleet_coordinator")

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
                self._converged.add(robot.id)
                self._node.get_logger().info(
                    f"coordinator_node: {robot.id}'s localization converged on "
                    "its seeded start"
                )

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
        places it within tolerance of its declared start pose. Otherwise a
        short string naming what's still wrong, for the throttled
        diagnostic log in _poll. Caller requires None to hold
        cfg.stable_polls polls in a row before trusting it.

        No transform-age check: AMCL republishes map->odom continuously
        (future-dated by transform_tolerance), so a stale *value* still
        carries a fresh stamp -- and comparing the stamp against this
        node's clock is a footgun when sim time and wall time disagree.
        The position check below is the real gate; a genuinely absent
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
        dx, dy, dyaw, blocker = convergence_offset(
            (t.x, t.y),
            yaw,
            (robot.start.x, robot.start.y, robot.start.theta),
            self._cfg,
        )
        # Signed samples (not abs) -- the timeout report needs the mean
        # direction to tell a systematic offset from scatter.
        self._offset_history[robot.id].append((dx, dy, dyaw))
        return blocker

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
