import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    declared_arguments = [
        DeclareLaunchArgument("run_rviz", default_value="true", description="Whether to run RViz"),
    ]

    run_rviz = LaunchConfiguration("run_rviz")

    demo_node = Node(
        package="crx_kinematics",
        executable="demo_node",
        name="demo_node",
        output="screen",
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        condition=IfCondition(run_rviz),
    )

    nodes_to_start = [demo_node, rviz_node]

    return LaunchDescription(declared_arguments + nodes_to_start)
