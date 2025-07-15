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
    param_file = os.path.join(pkg_share, 'config', 'referee_params.yaml')

    return LaunchDescription([
        Node(
            package='line_tracking_race_application',
            executable='referee_node',
            name='RefereeNode',
            namespace='line_tracking',
            parameters=[
                param_file
            ],
            output='screen'
        )
    ])
