"""
ROS2 PID Control Node for Line Tracking Robot

This module implements a PID controller for a differential drive robot performing
line tracking tasks. The controller receives error signals from a planning node
and outputs velocity commands to minimize tracking error.
"""

import os
import csv
from datetime import datetime
import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist
from ament_index_python.packages import get_package_share_directory

# Control constants
MAX_THRUST = 1.3      # Maximum forward velocity (m/s)
RAMP_UP = 0.5         # Thrust increment per control cycle for smooth acceleration

class ControlNode(Node):
    """
    ROS2 Node implementing PID control for line tracking.
    
    This node subscribes to error messages from a planning node and publishes
    velocity commands to control a differential drive robot. It implements
    PID control with configurable parameters and includes comprehensive logging
    for performance analysis.
    
    Attributes:
        max_duration (float): Maximum duration to run control loop (-1 for infinite)
        k_p (float): Proportional gain for PID controller
        k_i (float): Integral gain for PID controller  
        k_d (float): Derivative gain for PID controller
        setpoint (float): Desired error value (typically 0)
        prev_error (float): Previous error value for derivative calculation
        accumulated_integral (float): Accumulated integral term
        thrust (float): Current forward thrust value
        ISE (float): Integral of Squared Error performance metric
        started (bool): Flag indicating if control loop has started
        time_start (rclpy.Time): Timestamp when control started
        time_prev (rclpy.Time): Previous timestamp for dt calculation
    """
    
    def __init__(self):
        """Initialize the control node with parameters, publishers, subscribers, and logging."""
        super().__init__('control_node')

        # Declare and retrieve ROS2 parameters
        self._declare_parameters()
        self._get_parameters()
        
        self.get_logger().info(f"PID params: P={self.k_p}, I={self.k_i}, D={self.k_d}")

        # Initialize logging system
        self._setup_logging()

        # Initialize PID control variables
        self._initialize_control_variables()

        # Setup ROS2 communication
        self._setup_ros_communication()

        self.get_logger().info("Control node initialized successfully!")

    def _declare_parameters(self):
        """Declare ROS2 parameters with default values."""
        self.declare_parameter("duration", -1.0)  # -1 means no time limit
        self.declare_parameter("k_p", 0.01)       # Proportional gain
        self.declare_parameter("k_i", 0.00)       # Integral gain
        self.declare_parameter("k_d", 0.00)       # Derivative gain

    def _get_parameters(self):
        """Retrieve parameter values from ROS2 parameter server."""
        self.max_duration = self.get_parameter("duration").get_parameter_value().double_value
        self.k_p = self.get_parameter("k_p").get_parameter_value().double_value
        self.k_i = self.get_parameter("k_i").get_parameter_value().double_value
        self.k_d = self.get_parameter("k_d").get_parameter_value().double_value

    def _setup_logging(self):
        """Initialize CSV logging files for data collection and performance evaluation."""
        # Create timestamp for unique filenames
        date = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
        self.pkg_path = get_package_share_directory("line_tracking_race_application")
        
        # Open main data logging file
        self.open_logfile(date)
        
        # Open performance evaluation file
        self.open_performance_evaluation_file(date)
        self.errors = []
        self.times = []

    def _initialize_control_variables(self):
        """Initialize PID control and state variables."""
        # PID control variables
        self.setpoint = 0.0                    # Target error (0 = perfect line following)
        self.prev_error = 0.0                  # Previous error for derivative term
        self.accumulated_integral = 0.0        # Accumulated integral for I term
        self.thrust = 0.0                      # Current forward velocity
        self.ISE = 0.0                        # Integral of Squared Error metric
        
        # State management variables
        self.started = False                   # Control loop state flag
        self.time_start = None                # Start time for elapsed time calculation
        self.time_prev = None                 # Previous time for dt calculation

    def _setup_ros_communication(self):
        """Initialize ROS2 publishers and subscribers."""
        # Publisher for velocity commands to differential drive controller
        self.cmd_vel = self.create_publisher(Twist, "/car/cmd_vel", 10)

        # Subscriber for error signals from planning node
        self.error_sub = self.create_subscription(
            Float32, 
            "/planning/error", 
            self.handle_error_callback, 
            10
        )

    def handle_error_callback(self, msg):
        """
        Main PID control callback function.
        
        This function is called whenever a new error message is received from
        the planning node. It implements the PID control algorithm and publishes
        appropriate velocity commands.
        
        Args:
            msg (Float32): ROS2 message containing the tracking error
        """
        error = msg.data
        self.errors.append(error)
        time_now = self.get_clock().now()
        self.times.append(time_now.nanoseconds / 1e9)

        # Initialize timing on first callback
        if not self.started:
            self.time_start = time_now
            self.time_prev = time_now
            self.started = True
            return

        # Calculate elapsed time and time delta
        elapsed = (time_now - self.time_start).nanoseconds / 1e9
        dt = (time_now - self.time_prev).nanoseconds / 1e9

        # Check for duration timeout (if specified)
        if self.max_duration >= 0.0 and elapsed > self.max_duration:
            self.get_logger().warn("Maximum duration reached. Stopping robot.")
            self.stop()
            return

        # Skip control update if time delta is invalid
        if dt <= 0.0:
            return

        # Update performance metrics
        self._update_performance_metrics(error, dt)
        
        # Calculate PID control output
        control_output = self._calculate_pid_control(error, dt)
        
        # Update state variables for next iteration
        self.prev_error = error
        self.time_prev = time_now

        # Apply smooth thrust ramp-up for gentle acceleration
        self._update_thrust()

        # Generate velocity commands
        linear_x = self.thrust      # Forward velocity
        angular_z = control_output  # Angular velocity (steering)

        # Log data and publish commands
        self.log_data(elapsed, dt, error, control_output, linear_x, angular_z, 
                     self.k_p * error, 
                     self.k_i * self.accumulated_integral, 
                     self.k_d * (error - self.prev_error) / dt)
        self.publish_cmd_vel(linear_x, angular_z)

    def _update_performance_metrics(self, error, dt):
        """
        Update performance metrics for controller evaluation.
        
        Args:
            error (float): Current tracking error
            dt (float): Time delta since last update
        """
        # Calculate Integral of Squared Error using trapezoidal integration
        self.ISE += dt * (error**2 + self.prev_error**2) / 2.0

    def _calculate_pid_control(self, error, dt):
        """
        Calculate PID control output.
        
        Args:
            error (float): Current tracking error
            dt (float): Time delta since last update
            
        Returns:
            float: PID control output (angular velocity command)
        """
        # Update integral term using trapezoidal integration
        # Uses average of current and previous error, providing better approximation of the actual integral. 
        self.accumulated_integral += dt * (error + self.prev_error) / 2.0

        # Calculate individual PID terms
        p_term = self.k_p * error                                    # Proportional term
        i_term = self.k_i * self.accumulated_integral               # Integral term
        d_term = self.k_d * (error - self.prev_error) / dt         # Derivative term

        # Combine terms for final control output
        control_output = p_term + i_term + d_term
        
        return control_output

    def _update_thrust(self):
        """Apply smooth thrust ramp-up to prevent sudden acceleration."""
        if self.thrust < MAX_THRUST:
            self.thrust += RAMP_UP
            if self.thrust > MAX_THRUST:
                self.thrust = MAX_THRUST

    def publish_cmd_vel(self, linear_x, angular_z):
        """
        Publish velocity commands to the robot.
        
        Args:
            linear_x (float): Forward velocity command (m/s)
            angular_z (float): Angular velocity command (rad/s)
        """
        twist_msg = Twist()
        twist_msg.linear.x = linear_x
        twist_msg.angular.z = angular_z
        self.cmd_vel.publish(twist_msg)

    def open_logfile(self, date):
        """
        Open CSV file for logging control data.
        
        Args:
            date (str): Timestamp string for unique filename
        """
        # Create filename with PID parameters for easy identification
        pid_params = f"{self.k_p}-{self.k_i}-{self.k_d}".replace(".", ",")
        log_dir = os.path.join(self.pkg_path, "logs")
        os.makedirs(log_dir, exist_ok=True)
        filepath = os.path.join(log_dir, f"pid_log_{date}_[{pid_params}].csv")

        # Open file and setup CSV writer
        self.logfile = open(filepath, "w", newline="")
        self.log_writer = csv.writer(self.logfile)
        
        # Write header row
        self.log_writer.writerow([
            "Time", "dt", "Error", "CV", "LinearV", "AngularV", "P", "I", "D"
        ])

    def plot_error(self):
        plt.figure()
        plt.plot(self.times, self.errors, label="Tracking Error")
        plt.xlabel("Time (s)")
        plt.ylabel("Error")
        plt.title("Error Over Time")
        plt.grid(True)
        plt.legend()
        plt.show()

    def open_performance_evaluation_file(self, date):
        """
        Open CSV file for logging performance evaluation metrics.
        
        Args:
            date (str): Timestamp string for unique filename
        """
        # Create filename with PID parameters
        pid_params = f"{self.k_p}-{self.k_i}-{self.k_d}".replace(".", ",")
        eval_dir = os.path.join(self.pkg_path, "logs", "evaluations")
        os.makedirs(eval_dir, exist_ok=True)
        filepath = os.path.join(eval_dir, f"evaluation_{date}_[{pid_params}].csv")

        # Open file and setup CSV writer
        self.evaluation_file = open(filepath, "w", newline="")
        self.performance_index_writer = csv.writer(self.evaluation_file)
        
        # Write header row
        self.performance_index_writer.writerow(["ISE"])

    def log_data(self, elapsed, dt, error, control, linear_x, angular_z, p_term, i_term, d_term):
        """
        Log control data to CSV file.
        
        Args:
            elapsed (float): Elapsed time since start
            dt (float): Time delta since last update
            error (float): Current tracking error
            control (float): PID control output
            linear_x (float): Forward velocity command
            angular_z (float): Angular velocity command
            p_term (float): Proportional term value
            i_term (float): Integral term value
            d_term (float): Derivative term value
        """
        self.log_writer.writerow([
            elapsed, dt, error, control, linear_x, angular_z, p_term, i_term, d_term
        ])

    def log_performance_indices(self):
        """Log final performance metrics to evaluation file."""
        self.performance_index_writer.writerow([self.ISE])

    def stop(self):
        """
        Stop the robot and perform cleanup operations.
        
        This method is called when the control loop should terminate, either
        due to timeout or external interruption. It ensures the robot stops
        safely and logs are properly closed.
        """
        self.get_logger().info("Stopping robot...")

        # Create and publish zero velocity command to stop the robot
        twist_msg = Twist()
        twist_msg.linear.x = 0.0
        twist_msg.angular.z = 0.0

        # Publish multiple times to ensure message gets through
        for _ in range(10):
            self.cmd_vel.publish(twist_msg)

        # Log final performance metrics and close files
        self.log_performance_indices()
        self.logfile.close()
        self.evaluation_file.close()
        
        self.get_logger().info("Control node shutting down.")
        rclpy.shutdown()


def main(args=None):
    """
    Main function to initialize and run the control node.
    
    Args:
        args: Command line arguments (optional)
    """
    # Initialize ROS2 Python client library
    rclpy.init(args=args)
    
    # Create and start the control node
    node = ControlNode()
    
    try:
        # Run the node until interrupted
        rclpy.spin(node)
    except KeyboardInterrupt:
        # Handle graceful shutdown on Ctrl+C
        node.plot_error()
        node.stop()
    finally:
        # Cleanup resources
        if rclpy.ok():
            node.plot_error()

            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()