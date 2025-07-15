"""
ROS2 Line Tracking Planner Node

This module implements a computer vision-based line tracking planner for autonomous robots.
It processes camera images to detect lines/tracks and calculates error signals for the
control system. The node supports multiple planning strategies and error types.

The planner acts as the "eyes" of the robot, converting visual information into
numerical error signals that the PID controller can use for navigation.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
import sys

# Import custom planning strategy modules
# These modules contain the computer vision algorithms for line detection
from line_tracking.planning_strategies.centroid_strategy import CentroidStrategy
from line_tracking.planning_strategies.centerline_strategy import CenterlineStrategy
from line_tracking.planning_strategies.error_type import ErrorType


class PlannerNode(Node):
    """
    ROS2 Node for line tracking and path planning using computer vision.
    
    This node processes camera images to detect lines or tracks and calculates
    error signals for robot navigation. It supports different planning strategies
    and error calculation methods.
    
    The node architecture:
    Camera Image → Planning Strategy → Error Calculation → PID Controller
    
    Attributes:
        viz (bool): Flag to enable/disable visualization windows
        strategy (PlanningStrategy): Selected planning strategy instance
        camera_sub (Subscription): ROS2 subscriber for camera images
        error_pub (Publisher): ROS2 publisher for error signals
    """
    
    def __init__(self):
        """Initialize the planner node with configurable parameters and strategy selection."""
        super().__init__('planner_node')
        
        # Declare ROS2 parameters with default values
        self._declare_node_parameters()
        
        # Retrieve parameter values from ROS2 parameter server
        self._get_parameter_values()
        
        # Initialize the planning strategy based on parameters
        self._initialize_planning_strategy()
        
        # Setup ROS2 communication (publishers and subscribers)
        self._setup_ros_communication()
        
        self.get_logger().info("Planner node initialized successfully!")

    def _declare_node_parameters(self):
        """
        Declare ROS2 parameters that can be set via launch files or command line.
        
        Parameters:
            error_type (str): Type of error calculation ("offset" or "angle")
            viz (bool): Enable visualization windows for debugging
            strategy (str): Planning strategy to use ("centroid" or "centerline")
        """
        self.declare_parameter("error_type", "offset")    # Default to offset-based error
        self.declare_parameter("viz", False)              # Default to no visualization
        self.declare_parameter("strategy", "centroid")    # Default to centroid strategy

    def _get_parameter_values(self):
        """Retrieve and store parameter values from the ROS2 parameter server."""
        # Get error type parameter (how to calculate tracking error)
        self.error_type_arg = self.get_parameter("error_type").get_parameter_value().string_value
        
        # Get visualization flag (whether to show debug windows)
        self.viz = self.get_parameter("viz").get_parameter_value().bool_value
        
        # Get planning strategy parameter (which algorithm to use)
        self.planning_strategy_arg = self.get_parameter("strategy").get_parameter_value().string_value

    def _initialize_planning_strategy(self):
        """
        Initialize the selected planning strategy and error type.
        
        This method validates parameters and creates the appropriate strategy instance.
        If invalid parameters are provided, the node will shut down gracefully.
        """
        # Convert error type string to enum
        error_type = self._validate_and_convert_error_type()
        
        # Create strategy instance based on selected algorithm
        self._create_strategy_instance(error_type)

    def _validate_and_convert_error_type(self):
        """
        Validate and convert error type parameter to enum.
        
        Returns:
            ErrorType: Validated error type enum value
            
        Raises:
            SystemExit: If invalid error type is specified
        """
        if self.error_type_arg == "offset":
            # Offset error: lateral distance from line center
            return ErrorType.OFFSET
        elif self.error_type_arg == "angle":
            # Angle error: angular deviation from line direction
            return ErrorType.ANGLE
        else:
            # Invalid error type - log error and shutdown
            self.get_logger().fatal(
                f"Unknown error type '{self.error_type_arg}'. "
                f"Valid options are 'offset' or 'angle'. Shutting down."
            )
            sys.exit(1)

    def _create_strategy_instance(self, error_type):
        """
        Create and initialize the selected planning strategy.
        
        Args:
            error_type (ErrorType): Validated error type enum
            
        Raises:
            SystemExit: If invalid strategy is specified
        """
        if self.planning_strategy_arg == "centroid":
            # Centroid strategy: finds center of mass of detected line pixels
            self.strategy = CentroidStrategy(error_type, self.viz, self)
            self.get_logger().info("Using Centroid planning strategy")
            
        elif self.planning_strategy_arg == "centerline":
            # Centerline strategy: fits a line through detected features
            self.strategy = CenterlineStrategy(error_type, self.viz, self)
            self.get_logger().info("Using Centerline planning strategy")
            
        else:
            # Invalid strategy - log error and shutdown
            self.get_logger().fatal(
                f"Unknown strategy '{self.planning_strategy_arg}'. "
                f"Valid options are 'centroid' or 'centerline'. Shutting down."
            )
            sys.exit(1)

    def _setup_ros_communication(self):
        """Initialize ROS2 publishers and subscribers for communication."""
        # Subscriber for camera images
        # Topic: /car/camera/image_raw (raw camera feed from robot)
        # QoS: Queue depth of 10 (keeps last 10 messages if processing is slow)
        self.camera_sub = self.create_subscription(
            Image,                          # Message type
            '/car/camera/image_raw',        # Topic name
            self.camera_callback,           # Callback function
            10                              # QoS queue depth
        )
        
        # Publisher for calculated error signals
        # Topic: /planning/error (consumed by PID controller)
        # QoS: Queue depth of 10 for reliable delivery
        self.error_pub = self.create_publisher(Float32, '/planning/error', 10)
        
        self.get_logger().info("ROS2 communication setup complete")

    def camera_callback(self, msg: Image):
        """
        Process incoming camera images and publish tracking error.
        
        This is the main processing function that gets called whenever a new
        camera frame arrives. It performs the following steps:
        1. Pass image to selected planning strategy
        2. Calculate tracking error using computer vision
        3. Publish error signal for PID controller
        
        Args:
            msg (Image): ROS2 Image message containing camera frame
        """
        # Process the image using the selected strategy
        # This is where the computer vision magic happens
        calculated_error = self.strategy.plan(msg)
        
        # Check if strategy successfully calculated an error
        if calculated_error is None:
            # No line detected or processing failed
            # Don't publish anything - let controller handle missing data
            self.get_logger().debug("No error calculated - line not detected")
            return
        
        # Create and populate ROS2 message
        error_message = Float32()
        error_message.data = calculated_error
        
        # Publish error for PID controller to consume
        self.error_pub.publish(error_message)
        
        # Log for debugging (only in debug mode to avoid spam)
        self.get_logger().debug(f"Published error: {calculated_error:.3f}")


def main(args=None):
    """
    Main function to initialize and run the planner node.
    
    This function handles the complete lifecycle of the node:
    1. Initialize ROS2 system
    2. Create planner node instance
    3. Run the node (process messages)
    4. Handle graceful shutdown
    
    Args:
        args: Command line arguments (optional)
    """
    # Initialize ROS2 Python client library
    rclpy.init(args=args)
    
    # Initialize planner node variable for cleanup
    planner_node = None
    
    try:
        # Create the planner node instance
        # This calls __init__ which sets up everything
        planner_node = PlannerNode()
        
        # Start the ROS2 event loop
        # This blocks and processes incoming messages until interrupted
        rclpy.spin(planner_node)
        
    except (KeyboardInterrupt, SystemExit) as e:
        # Handle different types of shutdown gracefully
        if isinstance(e, SystemExit):
            # SystemExit raised during initialization (invalid parameters)
            print("Planner node failed to initialize and is shutting down.")
        else:
            # KeyboardInterrupt (Ctrl+C) - normal shutdown
            print("Planner node shutting down gracefully.")
            
    finally:
        # Cleanup resources regardless of how we got here
        # This ensures proper cleanup even if something goes wrong
        if rclpy.ok():
            # Check if node was successfully created before destroying it
            if planner_node is not None:
                try:
                    planner_node.destroy_node()
                    print("Planner node destroyed successfully.")
                except Exception as e:
                    print(f"Error destroying planner node: {e}")
            
            # Shutdown ROS2 system
            rclpy.shutdown()
            print("ROS2 system shutdown complete.")


# Standard Python entry point
if __name__ == "__main__":
    main()