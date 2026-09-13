"""Launch file for fleet_coordinator's coordinator_node.

Covers the four arguments actually worth varying run to run (mission_file,
use_sim_time, spooky.map_id, initial_pose.publish). Everything else
coordinator_node.py exposes as a ROS2 parameter (stall.*, deadlock.*,
convergence.*, ordering.*, dispatch.*, recovery.*, the rest of spooky.*
-- see its own declare_dataclass_parameters) is tuned rarely enough that
one launch argument each would just be noise; override those via
params_file instead (see config/params.example.yaml for the full list
and current defaults). params_file is applied first, so the four
explicit arguments below always win if a params_file also sets one of
them.

Usage:
    ros2 launch fleet_coordinator coordinator.launch.py \\
        mission_file:=config/sim_order.yaml use_sim_time:=true \\
        spooky.map_id:=test-scenario_simple initial_pose.publish:=true
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    mission_file = LaunchConfiguration("mission_file")
    use_sim_time = LaunchConfiguration("use_sim_time")
    map_id = LaunchConfiguration("spooky.map_id")
    initial_pose_publish = LaunchConfiguration("initial_pose.publish")
    params_file = LaunchConfiguration("params_file")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "mission_file",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("fleet_coordinator"),
                        "config",
                        "mission.example.yaml",
                    ]
                ),
                description=(
                    "Fleet mission YAML -- per-robot start/goal/priority/"
                    "robot_radius/... (Fleet.from_yaml's format)."
                ),
            ),
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="false",
                description=(
                    "Use /clock instead of wall time -- true for sim, false "
                    "(the ROS2-wide default) on real hardware."
                ),
            ),
            DeclareLaunchArgument(
                "spooky.map_id",
                default_value="no_obs5x5",
                description=(
                    "Spooky map id -- must be a key GET /v1/maps actually "
                    "lists on the target server (see spooky_client.py)."
                ),
            ),
            DeclareLaunchArgument(
                "initial_pose.publish",
                default_value="false",
                description=(
                    "Seed AMCL from the mission file's declared start and "
                    "gate planning on convergence. Correct only when that "
                    "declared start is a trustworthy measurement, not an "
                    "assumption -- true in sim (you control both); on real "
                    "hardware only with an independently accurate declared "
                    "start (known docks, ...). See convergence.ConvergenceGate."
                ),
            ),
            DeclareLaunchArgument(
                "params_file",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("fleet_coordinator"),
                        "config",
                        "params.example.yaml",
                    ]
                ),
                description=(
                    "ROS2 params YAML for every tunable besides the four "
                    "above -- see config/params.example.yaml for the full "
                    "list. Empty by default (that file's own defaults)."
                ),
            ),
            Node(
                package="fleet_coordinator",
                executable="coordinator_node",
                name="fleet_coordinator",
                output="screen",
                parameters=[
                    params_file,
                    {
                        "mission_file": mission_file,
                        "use_sim_time": use_sim_time,
                        "spooky.map_id": map_id,
                        "initial_pose.publish": initial_pose_publish,
                    },
                ],
            ),
        ]
    )
