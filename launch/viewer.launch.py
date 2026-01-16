from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node

def generate_launch_description():

    return LaunchDescription([

        # RViz2 with correct config
        ExecuteProcess(
            cmd=['rviz2', '-d', '/home/remon/ros2_ws/src/realsense_wifi/config/realsense_view.rviz'],
            output='screen'
        )
    ])
