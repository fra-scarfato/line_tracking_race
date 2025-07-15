import os
from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node


def generate_launch_description():
    # Get the package share directory
    pkg_share = get_package_share_directory('line_tracking_race_application')

    # Construct the path to the parameter file
    param_file = os.path.join(pkg_share, 'config', 'pid_params.yaml')

    return LaunchDescription([
        DeclareLaunchArgument(
            'duration',
            default_value='-1.0',
            description='How long the node should run for. -1.0 indicates no limit.'
        ),

        Node(
            package='line_tracking_race_application',
            executable='control_node',
            name='ControlNode',
            namespace='line_tracking',
            parameters=[
                {'duration': LaunchConfiguration('duration')},
                param_file
            ],
            output='screen'
        )
    ])
