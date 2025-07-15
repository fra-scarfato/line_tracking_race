from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Declare launch arguments
    viz_arg = DeclareLaunchArgument(
        'viz',
        default_value='False',
        description='Whether to visualize debugging data'
    )
    strategy_arg = DeclareLaunchArgument(
        'strategy',
        default_value='centroid',
        description='Planning strategy for waypoint selection'
    )
    error_type_arg = DeclareLaunchArgument(
        'error_type',
        default_value='angle',
        description='Type of waypoint error to compute'
    )

    # Create the planner node
    planner_node = Node(
        package='line_tracking_race_application',
        executable='planner_node',
        namespace='line_tracking',
        name='PlannerNode',
        parameters=[{
            'viz': LaunchConfiguration('viz'),
            'strategy': LaunchConfiguration('strategy'),
            'error_type': LaunchConfiguration('error_type'),
        }]
    )

    return LaunchDescription([
        viz_arg,
        strategy_arg,
        error_type_arg,
        planner_node,
    ])
