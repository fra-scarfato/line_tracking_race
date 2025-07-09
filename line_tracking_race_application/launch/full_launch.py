from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Get package paths
    app_pkg = get_package_share_directory('line_tracking_race_application')
    sim_pkg = get_package_share_directory('line_tracking_race_bringup')

    # Launch files
    planner_launch = os.path.join(app_pkg, 'launch', 'planner_launch.py')
    control_launch = os.path.join(app_pkg, 'launch', 'control_launch.py')
    race_launch = os.path.join(sim_pkg, 'launch', 'line_tracking_race.launch.py')

    return LaunchDescription([
        # Declare arguments
        DeclareLaunchArgument('viz', default_value='False'),
        DeclareLaunchArgument('duration', default_value='-1.0'),
        DeclareLaunchArgument('strategy', default_value='centroid'),
        DeclareLaunchArgument('error_type', default_value='angle'),
        
        # Include planner.launch.py
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(planner_launch),
            launch_arguments={
                'viz': LaunchConfiguration('viz'),
                'strategy': LaunchConfiguration('strategy'),
                'error_type': LaunchConfiguration('error_type')
            }.items()
        ),

        # Include control.launch.py
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(control_launch),
            launch_arguments={
                'duration': LaunchConfiguration('duration'),
            }.items()
        ),

         # Include simulation launch
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(race_launch)
        ),

        
    ])
