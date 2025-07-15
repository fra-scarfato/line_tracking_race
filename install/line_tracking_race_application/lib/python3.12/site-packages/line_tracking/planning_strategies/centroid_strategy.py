"""
Centroid Strategy Module for Line Tracking

This module implements a simple planning strategy for autonomous line tracking
based on the centroid of a detected yellow track region. It calculates the
centroid of the track in each frame and computes a navigation error (either
offset or angle) relative to the vehicle's position.

"""

import math
import numpy as np

import cv2 as cv
from cv_bridge import CvBridge

import rclpy
from rclpy.node import Node

from line_tracking.planning_strategies.error_type import ErrorType
from line_tracking.visualizer import Visualizer

# OpenCV hue value constraints
MAX_HUE = 179  # In OpenCV, hue ranges from 0 to 179 (not 0–360 like standard HSV)

# HSV color space thresholds for yellow track detection
# Format: (Hue, Saturation, Value)
LOWER_YELLOW = (20, 50, 50)   # Lower bound for yellow detection
UPPER_YELLOW = (30, 255, 255) # Upper bound for yellow detection


class CentroidStrategy:
    """
    A planning strategy based on the centroid of the detected track.

    This class uses basic computer vision techniques to find the centroid of
    yellow-colored track regions in the camera image. The centroid is used as
    the next waypoint, and the strategy computes either an offset or angular
    error to guide vehicle steering.

    Attributes:
        error_type (ErrorType): The type of error calculation to use (OFFSET or ANGLE)
        node (Node): ROS2 node for logging and communication
        viz (Visualizer): Optional visualizer for debug display
        cv_bridge (CvBridge): Bridge for converting ROS image messages to OpenCV format
        prev_centroid (tuple): Previous centroid position for fallback
    """

    def __init__(self, error_type, should_visualize, node):
        """
        Initialize the centroid strategy.

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

        # Initialize ROS-OpenCV bridge
        self.cv_bridge = CvBridge()

        # Fallback centroid if detection fails
        self.prev_centroid = (0, 0)

    def plan(self, img_msg):
        """
        Process an image message to detect the track centroid and return waypoint error.

        This function:
        1. Converts the ROS image to OpenCV format
        2. Applies a yellow mask in HSV space
        3. Computes the centroid of the mask
        4. Calculates error based on centroid position

        Args:
            img_msg: ROS image message containing the camera feed

        Returns:
            float: Normalized error based on offset or angle from the centroid
        """
        image = self.cv_bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
        height, width, _ = image.shape

        # Convert image to HSV and extract yellow areas
        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, np.array(LOWER_YELLOW), np.array(UPPER_YELLOW))

        # Compute centroid using image moments
        M = cv.moments(mask)
        if M["m00"] != 0:
            centroid = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            self.prev_centroid = centroid
        else:
            self.node.get_logger().warn("No centroid found, reusing previous waypoint.")
            centroid = self.prev_centroid

        # Crosshair represents vehicle forward direction (center of the image)
        crosshair = (math.floor(width / 2), math.floor(height / 2))

        # Position is assumed to be bottom-center of the image
        position = (math.floor(width / 2), height - 1)

        # Compute error based on selected strategy
        if self.error_type == ErrorType.OFFSET:
            err, offset = self.compute_offset_error(centroid, crosshair, width / 2)
        elif self.error_type == ErrorType.ANGLE:
            err, angle = self.compute_angle_error(centroid, position)
        else:
            self.node.get_logger().error("Unknown error type. Exiting.")
            rclpy.shutdown()

        # Visualization for debugging
        if self.viz is not None:
            self.viz.build_basic_bg(image)

            if self.error_type == ErrorType.OFFSET:
                self.viz.build_offset_error_overlay(crosshair, centroid)
            elif self.error_type == ErrorType.ANGLE:
                self.viz.build_angle_error_overlay(crosshair, centroid, position, angle)
            else:
                self.node.get_logger().error("Unknown error type. Exiting.")
                rclpy.shutdown()

            self.viz.show()

        return err

    def compute_offset_error(self, waypoint, crosshair, max_offset):
        """
        Compute normalized horizontal offset error between crosshair and centroid.

        Args:
            waypoint (tuple): Target point (centroid), given as (x, y) pixel coordinates.
            crosshair (tuple): Center of screen (or reference point), also (x, y).
            max_offset (float): Maximum expected pixel offset from center (for normalization).

        Returns:
            tuple: 
                normalized_error (float): Value in range [-1, 1] representing how far 
                                        the waypoint is from the crosshair horizontally.
                raw_offset (float): Raw pixel offset between waypoint and crosshair (x-axis).
        """
        # Calculate the horizontal offset between the waypoint and the crosshair
        offset = crosshair[0] - waypoint[0]

        # Normalize the offset to range [-1, 1], where 0 means centered
        normalized_error = (offset + max_offset) / max_offset - 1

        return normalized_error, offset


    def compute_angle_error(self, waypoint, position):
        """
        Compute normalized angular error between vehicle heading and centroid direction.

        Args:
            waypoint (tuple): Target point (centroid), as (x, y) pixel coordinates.
            position (tuple): Vehicle's current position, typically bottom-center of the image (x, y).

        Returns:
            tuple:
                normalized_error (float): Value in range [-1, 1], representing angular deviation.
                raw_angle_degrees (float): Actual angle (in degrees) between vehicle and waypoint direction.
        """
        # Calculate Euclidean distance between vehicle position and waypoint
        dist = math.sqrt((waypoint[0] - position[0]) ** 2 + (waypoint[1] - position[1]) ** 2)

        # Compute the angle between the line to the waypoint and vertical axis
        # (x-difference / hypotenuse) gives sin(theta); result in radians
        angle = math.asin((position[0] - waypoint[0]) / dist)

        # Convert angle from radians to degrees
        angle_deg = angle * 180 / math.pi

        # Normalize angle to [-1, 1]; -90° becomes -1, 0° becomes 0, +90° becomes +1
        normalized_error = (angle_deg + 90) / 90 - 1

        return normalized_error, angle_deg
