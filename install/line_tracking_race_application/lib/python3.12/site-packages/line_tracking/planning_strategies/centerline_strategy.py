"""
Centerline Strategy Module for Line Tracking

This module implements a planning strategy for autonomous line tracking that
revolves around finding the track's centerline and choosing waypoints along it.
The strategy uses computer vision to detect yellow track markers and computes
navigation errors for path following.

"""

import math
import numpy as np

import cv2 as cv
from cv_bridge import CvBridge

import rclpy
from rclpy.node import Node

from line_tracking.planning_strategies.error_type import ErrorType
from line_tracking.visualizer import Visualizer
import line_tracking.colors as colors

# OpenCV hue value constraints
MAX_HUE = 179  # In OpenCV, hue ranges from 0 to 179 (not 0-360 like standard HSV)

# HSV color space thresholds for yellow track detection
# Format: (Hue, Saturation, Value)
LOWER_YELLOW = (20, 50, 50)   # Lower bound for yellow detection
UPPER_YELLOW = (30, 255, 255) # Upper bound for yellow detection


class CenterlineStrategy:
    """
    A planning strategy for autonomous line tracking based on centerline detection.
    
    This class implements a computer vision-based approach to track following where
    the system detects yellow track boundaries, computes the centerline between them,
    and selects waypoints for navigation. It supports both offset-based and angle-based
    error computation methods.
    
    Attributes:
        error_type (ErrorType): The type of error calculation to use (OFFSET or ANGLE)
        node (Node): ROS2 node for logging and communication
        viz (Visualizer): Optional visualizer for debug display
        cv_bridge (CvBridge): Bridge for converting ROS image messages to OpenCV format
        prev_offset (float): Previous offset value for fallback scenarios
        prev_waypoint (tuple): Previous waypoint coordinates for fallback scenarios
    """
    
    def __init__(self, error_type, should_visualize, node):
        """
        Initialize the centerline strategy.
        
        Args:
            error_type (ErrorType): Type of offset error to compute (OFFSET or ANGLE)
            should_visualize (bool): Whether to visualize debug data in a separate window
            node (Node): ROS2 node instance for logging and communication
        """
        self.error_type = error_type
        self.node = node
        
        # Initialize visualizer if requested
        if should_visualize:
            self.viz = Visualizer()
        else:
            self.viz = None

        # Initialize ROS-OpenCV bridge for image conversion
        self.cv_bridge = CvBridge()
        
        # Initialize fallback values for when track detection fails
        self.prev_offset = 0
        self.prev_waypoint = (0, 0)

    def plan(self, img_msg):
        """
        Apply the centerline strategy to process an image and return waypoint error.
        
        This is the main processing function that:
        1. Converts ROS image message to OpenCV format
        2. Detects track boundaries
        3. Computes centerline
        4. Selects waypoint
        5. Calculates navigation error
        
        Args:
            img_msg: ROS image message containing the camera feed
            
        Returns:
            float: Waypoint error based on the configured error type (offset or angle)
        """
        # Convert ROS image message to OpenCV format
        image = self.cv_bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
        height, width, _ = image.shape

        # Extract track outline from the image
        track_outline = self.get_track_outline(image)

        # Crop image to focus on relevant area and remove border artifacts
        # - Remove top half to focus on nearby track sections
        # - Remove 100 pixels from left/right borders to eliminate edge artifacts
        cropped_outline = track_outline[
            int(height / 2) : (height - 10),  # Vertical crop: middle to bottom-10
            100 : (width - 100)              # Horizontal crop: remove 100px borders
        ]
        cr_height, cr_width = cropped_outline.shape

        # Extract left and right track boundaries
        left_limit, right_limit = self.extract_track_limits(cropped_outline)
        
        # Compute centerline between the track boundaries
        centerline = self.compute_centerline(left_limit, right_limit)

        # Define reference points for navigation
        crosshair = (math.floor(cr_width / 2), math.floor(cr_height / 2))  # Screen center
        position = (math.floor(cr_width / 2), cr_height - 1)               # Vehicle position (bottom center)

        # Handle track detection failure by using previous values
        if left_limit.size == 0 or right_limit.size == 0:
            self.node.get_logger().warn("Can't compute centerline, reusing previous waypoint.")
            waypoint = self.prev_waypoint
            waypoint_offset = self.prev_offset
        else:
            # Select next waypoint and compute its offset
            waypoint, waypoint_offset = self.get_next_waypoint(centerline, crosshair)
            
            # Store values for potential future fallback
            self.prev_waypoint = waypoint
            self.prev_offset = waypoint_offset

        # Compute navigation error based on configured error type
        if self.error_type == ErrorType.OFFSET:
            err, offset = self.compute_offset_error(waypoint, crosshair, cr_width / 2)
        elif self.error_type == ErrorType.ANGLE:
            err, angle = self.compute_angle_error(waypoint, position)
        else:
            self.node.get_logger().error(f"Unknown error type. Exiting")
            rclpy.shutdown()

        # Generate debug visualization if enabled
        if self.viz is not None:
            # Build base visualization showing track and centerline
            self.viz.build_track_bg(
                cr_height, cr_width, left_limit, right_limit, centerline
            )

            # Add error-specific visualization overlay
            if self.error_type == ErrorType.OFFSET:
                self.viz.build_offset_error_overlay(crosshair, waypoint)
            elif self.error_type == ErrorType.ANGLE:
                self.viz.build_angle_error_overlay(crosshair, waypoint, position, angle)
            else:
                self.node.get_logger().error(f"Unknown error type. Exiting")
                rclpy.shutdown()

            # Display the visualization
            self.viz.show()

        return err

    def get_track_outline(self, input):
        """
        Detect the track in the input image and return its contour outline.
        
        This method uses HSV color space filtering to isolate yellow track markers,
        then finds contours and draws them on a binary image.
        
        Args:
            input (np.ndarray): Input BGR image from camera
            
        Returns:
            np.ndarray: Binary image with track outline drawn in magenta
        """
        height, width, _ = input.shape
        
        # Create empty binary image for track outline
        track_outline = np.zeros((height, width), dtype=np.uint8)

        # Convert BGR to HSV color space for better color-based segmentation
        hsv = cv.cvtColor(input, cv.COLOR_BGR2HSV)
        
        # Create binary mask for yellow track markers
        mask = cv.inRange(hsv, np.array(LOWER_YELLOW), np.array(UPPER_YELLOW))

        # Find contours in the binary mask
        contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        
        # Draw the first (largest) contour on the output image
        # Note: This assumes the track is the largest yellow object in the image
        if contours:
            cv.drawContours(track_outline, contours, 0, colors.MAGENTA)

        return track_outline

    def extract_track_limits(self, track_outline):
        """
        Extract left and right track boundaries from the track outline image.
        
        This method uses connected components analysis to separate the track outline
        into distinct left and right boundaries.
        
        Args:
            track_outline (np.ndarray): Binary image containing track outline
            
        Returns:
            tuple: (left_limit, right_limit) - Two numpy arrays containing 
                   coordinates of left and right track boundaries respectively
        """
        # Use connected components to separate track boundaries
        # Expected labels: 0=background, 1=left boundary, 2=right boundary
        _, labels = cv.connectedComponents(track_outline)

        # Extract left track boundary points
        left_limit_cols, left_limit_rows = np.where(labels == 1)
        left_limit = np.column_stack((left_limit_rows, left_limit_cols))[
            ::10  # Subsample every 10th point to reduce computational load
        ]

        # Extract right track boundary points  
        right_limit_cols, right_limit_rows = np.where(labels == 2)
        right_limit = np.column_stack((right_limit_rows, right_limit_cols))[
            ::10  # Subsample every 10th point to reduce computational load
        ]

        return left_limit, right_limit

    def compute_centerline(self, left, right):
        """
        Compute the centerline between left and right track boundaries.
        
        This method calculates the midpoint between corresponding points on the
        left and right track boundaries to create a centerline trajectory.
        
        Args:
            left (np.ndarray): Array of (x, y) coordinates for left track boundary
            right (np.ndarray): Array of (x, y) coordinates for right track boundary
            
        Returns:
            np.ndarray: Array of (x, y) coordinates representing the centerline
        """
        centerline = []
        
        # Calculate midpoint between each pair of corresponding left/right points
        for (x1, y1), (x2, y2) in zip(left, right):
            # Compute centerpoint coordinates
            xc = math.floor((x1 + x2) / 2)
            yc = math.floor((y1 + y2) / 2)
            
            centerline.append((xc, yc))

        return np.array(centerline)

    def get_next_waypoint(self, trajectory, crosshair):
        """
        Select the next waypoint from the trajectory based on crosshair position.
        
        This method finds the closest point on the centerline trajectory that is
        ahead of the vehicle (above the crosshair in image coordinates).
        
        Args:
            trajectory (np.ndarray): Array of centerline points
            crosshair (tuple): (x, y) coordinates of screen center
            
        Returns:
            tuple: (waypoint, x_offset) where waypoint is (x, y) coordinates 
                   and x_offset is the horizontal offset from crosshair
        """
        # Handle empty trajectory case
        if trajectory.size == 0:
            return crosshair, 0

        center_x, center_y = crosshair

        # Find the closest valid waypoint
        closest = 0
        closest_dist = float("inf")
        
        for i, (x, y) in enumerate(trajectory):
            # Skip waypoints that are too close to or behind the crosshair
            # (30 pixel buffer to avoid selecting waypoints too close to vehicle)
            if y > center_y - 30:
                continue

            # Calculate Euclidean distance to crosshair
            dist = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            
            # Update closest waypoint if this one is closer
            if dist < closest_dist:
                closest_dist = dist
                closest = i

        # Return the closest waypoint and its horizontal offset
        waypoint = trajectory[closest]
        x_offset = waypoint[0] - center_x
        
        return waypoint, x_offset

    def compute_offset_error(self, waypoint, crosshair, max_offset):
        """
        Compute the horizontal offset error to the waypoint.
        
        This method calculates the horizontal distance between the crosshair
        and waypoint, then normalizes it to the [-1, 1] range.
        
        Args:
            waypoint (tuple): (x, y) coordinates of target waypoint
            crosshair (tuple): (x, y) coordinates of screen center
            max_offset (float): Maximum possible offset value for normalization
            
        Returns:
            tuple: (normalized_error, raw_offset) where normalized_error is in [-1, 1]
                   and raw_offset is the actual pixel offset
        """
        # Calculate raw horizontal offset
        offset =  crosshair[0] - waypoint[0] 
        
        # Normalize offset to [-1, 1] range
        # Formula: (offset + max_offset) / (2 * max_offset) * 2 - 1
        # Simplified to: (offset + max_offset) / max_offset - 1
        normalized_error = (offset + max_offset) / max_offset - 1
        
        return normalized_error, offset

    def compute_angle_error(self, waypoint, position):
        """
        Compute the angular error to the waypoint.
        
        This method calculates the angle between the vehicle's forward direction
        (vertical line) and the line connecting the vehicle to the waypoint,
        then normalizes it to the [-1, 1] range.
        
        Args:
            waypoint (tuple): (x, y) coordinates of target waypoint
            position (tuple): (x, y) coordinates of vehicle position
            
        Returns:
            tuple: (normalized_error, raw_angle) where normalized_error is in [-1, 1]
                   and raw_angle is the actual angle in degrees
        """
        # Calculate distance to waypoint
        dist = math.sqrt(
            (waypoint[0] - position[0]) ** 2 + (waypoint[1] - position[1]) ** 2
        )
        
        # Calculate angle using arcsine (horizontal displacement / total distance)
        # This gives the angle from the vertical (forward direction)
        angle = math.asin((position[0] - waypoint[0]) / dist)
        
        # Convert from radians to degrees
        angle_deg = angle * 180 / math.pi
        
        # Normalize angle from [-90, 90] degrees to [-1, 1] range
        # Formula: (angle + 90) / 180 * 2 - 1
        # Simplified to: (angle + 90) / 90 - 1
        normalized_error = (angle_deg + 90) / 90 - 1
        
        return normalized_error, angle_deg