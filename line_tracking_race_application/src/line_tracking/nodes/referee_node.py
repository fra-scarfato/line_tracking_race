import json
import os
import math
import rclpy
import numpy as np

from rclpy.node import Node
from rclpy.time import Time
from nav_msgs.msg import Odometry
from std_msgs.msg import String

from ament_index_python.packages import get_package_share_directory
from line_tracking.track_utility import TrackUtility

# === Configuration Constants ===
DEFAULT_MIN_LAP_TIME = 5.0
FINISH_LINE_THRESHOLD = 2.0  # meters - used to detect crossing of the finish line (near s=0)
INTERPOLATION_ENABLED = True  # Enables sub-sample precision lap timing
SPEED_CALCULATION_ENABLED = True  # Toggle speed tracking

class LapTimer:
    """
    Tracks the lap count and computes lap durations using the curvilinear
    coordinate `s`. Optionally uses interpolation for higher time precision.
    """
    def __init__(self, s0, L, threshold=FINISH_LINE_THRESHOLD):
        self.s0 = s0              # Finish line reference (usually s=0)
        self.L = L                # Total track length (for wraparound handling)
        self.threshold = threshold
        self.prev_s = None
        self.prev_timestamp = None
        self.laps = 0
        self.last_crossing_s = None
        self.last_crossing_time = None
        self.position_history = []  # Recent (s, time) pairs for interpolation
        self.max_history_size = 10

    def update(self, s_car, timestamp):
        """
        Main update function to detect lap completions.
        Returns:
            - lap_completed: bool
            - precise_crossing_time: float (if available)
            - crossing_s: float (usually 0.0 at finish line)
        """
        self.position_history.append((s_car, timestamp))
        if len(self.position_history) > self.max_history_size:
            self.position_history.pop(0)

        lap_completed = False
        precise_crossing_time = None
        crossing_s = None

        if self.prev_s is not None:
            if self._detect_lap_crossing(self.prev_s, s_car):
                self.laps += 1
                lap_completed = True
                # Refined crossing time using interpolation
                if INTERPOLATION_ENABLED and len(self.position_history) >= 2:
                    precise_crossing_time, crossing_s = self._interpolate_crossing_time()
                else:
                    precise_crossing_time = timestamp
                    crossing_s = s_car

                self.last_crossing_s = crossing_s
                self.last_crossing_time = precise_crossing_time

        self.prev_s = s_car
        self.prev_timestamp = timestamp

        return lap_completed, precise_crossing_time, crossing_s

    def _detect_lap_crossing(self, prev_s, curr_s):
        """
        Detect whether the finish line was crossed in forward direction.
        Handles wraparound from near `L` to near 0.
        """
        if abs(curr_s - prev_s) > self.L / 2:
            # Wrapped around
            if prev_s > self.L - self.threshold and curr_s < self.threshold:
                return True  # Forward crossing
            elif prev_s < self.threshold and curr_s > self.L - self.threshold:
                return False  # Likely reversed
        elif prev_s > self.threshold and curr_s <= self.threshold:
            return True

        return False

    def _interpolate_crossing_time(self):
        """
        Linearly interpolate the time when the car crossed s=0
        using the last two points in history.
        """
        for i in range(len(self.position_history) - 1):
            s1, t1 = self.position_history[i]
            s2, t2 = self.position_history[i + 1]

            if self._crossing_between_points(s1, s2):
                # Handle wraparound for interpolation
                if abs(s2 - s1) > self.L / 2:
                    if s1 > s2:
                        s1_adj = s1 - self.L
                        alpha = -s1_adj / (s2 - s1_adj)
                    else:
                        s2_adj = s2 - self.L
                        alpha = -s1 / (s2_adj - s1)
                else:
                    alpha = -s1 / (s2 - s1)

                alpha = max(0.0, min(1.0, alpha))
                crossing_time = t1 + alpha * (t2 - t1)
                return crossing_time, 0.0  # Crossed exactly at s = 0

        return self.position_history[-1][1], self.position_history[-1][0]

    def _crossing_between_points(self, s1, s2):
        """Check if s wrapped over finish line between s1 and s2"""
        if abs(s2 - s1) > self.L / 2:
            return s1 > self.L - self.threshold and s2 < self.threshold
        return s1 > self.threshold and s2 <= self.threshold

class RefereeNode(Node):
    def __init__(self):
        super().__init__('referee_node')

        # State variables
        self.previous_position = None
        self.prev_s = None
        self.prev_timestamp = None
        self.session_start_time = None
        self.last_lap_time = None
        self.lap_counter = 0
        self.lap_durations = []
        self.total_distance = 0.0
        self.instantaneous_speeds = []

        # Track setup
        self.track_utility = TrackUtility()
        self.track_pts = np.array(self.track_utility.get_world_points())
        self.track_s = np.array(self.track_utility.get_curvilinear_abscissa())
        self.L = float(self.track_s[-1])  # Total track length
        self.s0 = 0.0  # Start line position
        self.lap_timer = LapTimer(s0=self.s0, L=self.L)

        # ROS communication
        self.create_subscription(Odometry, '/car/odom', self._on_odometry_received, 10)
        self.stats_pub = self.create_publisher(String, '/referee/stats', 10)
        self.lap_times_pub = self.create_publisher(String, '/referee/lap_times', 10)

        self.create_timer(0.5, self.publish_stats)
        self.create_timer(1.0, self.publish_lap_times)

        self.get_logger().info(f"Referee initialized: track length={self.L:.2f}m")

    def publish_stats(self):
        now = self.get_clock().now().nanoseconds / 1e9

        best = min(self.lap_durations) if self.lap_durations else 0.0
        avg = sum(self.lap_durations) / len(self.lap_durations) if self.lap_durations else 0.0

        stats = {
            'lap_counter': self.lap_counter,
            'best_lap_time': best,
            'average_lap_time': avg,
            'total_distance': self.total_distance / 10,  # optional scaling
        }

        msg = String()
        msg.data = json.dumps(stats)
        self.stats_pub.publish(msg)

    def publish_lap_times(self):
        msg = String()
        msg.data = json.dumps(self.lap_durations)
        self.lap_times_pub.publish(msg)


    def _get_yaw_from_quaternion(self, o):
        """
        Convert quaternion orientation into yaw (2D heading).
        Using standard formula: yaw = atan2(2(wz + xy), 1 - 2(y² + z²))
        """
        return (lambda a: (a + math.pi) % (2*math.pi) - math.pi)(
            math.atan2(2.0 * (o.w * o.z + o.x * o.y), 1.0 - 2.0 * (o.y*o.y + o.z*o.z))
        )

 
    def _xy_to_s(self, x, y):
        """
        Project (x, y) onto the closest point on the centerline,
        and interpolate s for smoother estimation.
        """
        d2 = np.sum((self.track_pts - np.array([x, y]))**2, axis=1)
        idx = int(np.argmin(d2))

        if 0 < idx < len(self.track_pts) - 1:
            p_prev = self.track_pts[idx - 1]
            p_curr = self.track_pts[idx]
            p_next = self.track_pts[idx + 1]

            d_prev = np.linalg.norm([x - p_prev[0], y - p_prev[1]])
            d_next = np.linalg.norm([x - p_next[0], y - p_next[1]])
            d_curr = np.linalg.norm([x - p_curr[0], y - p_curr[1]])

            if d_prev < d_next and d_prev < d_curr:
                alpha = d_prev / (d_prev + d_curr) if (d_prev + d_curr) > 0 else 0
                s_interp = self.track_s[idx - 1] * (1 - alpha) + self.track_s[idx] * alpha
            elif d_next < d_curr:
                alpha = d_next / (d_next + d_curr) if (d_next + d_curr) > 0 else 0
                s_interp = self.track_s[idx] * (1 - alpha) + self.track_s[idx + 1] * alpha
            else:
                s_interp = float(self.track_s[idx])
        else:
            s_interp = float(self.track_s[idx])

        return s_interp

    # --- Calculate curvilinear speed (ds/dt), handling wrap-around ---
    def _calculate_curvilinear_speed(self, s_curr, s_prev, dt):
        if dt <= 0:
            return 0.0
        ds = s_curr - s_prev
        if abs(ds) > self.L / 2:
            if ds > 0:
                ds = ds - self.L  # Wrapped around backwards
            else:
                ds = ds + self.L
        return abs(ds) / dt

    # --- Handle odometry messages from the vehicle ---
    def _on_odometry_received(self, msg: Odometry):
        pos = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        t = Time.from_msg(msg.header.stamp).nanoseconds / 1e9

        if self.previous_position is None:
            self.session_start_time = t
            self.previous_position = pos
            self.prev_s = self._xy_to_s(*pos)
            self.prev_timestamp = t
            return

        # Step distance traveled
        step = math.hypot(pos[0] - self.previous_position[0],
                          pos[1] - self.previous_position[1])
        self.total_distance += step

        # Project to curvilinear s
        s_cur = self._xy_to_s(*pos)

        # Estimate speed (ds/dt)
        if self.prev_s is not None and self.prev_timestamp is not None:
            dt = t - self.prev_timestamp
            speed = self._calculate_curvilinear_speed(s_cur, self.prev_s, dt)
            self.instantaneous_speeds.append(speed)

        # Lap detection logic
        lap_completed, precise_crossing_time, crossing_s = self.lap_timer.update(s_cur, t)

        if lap_completed:
            if self.lap_timer.laps == 1:
                self.get_logger().info("First crossing detected → starting lap timer.")
                self.last_lap_time = precise_crossing_time
            else:
                lap_time = precise_crossing_time - self.last_lap_time
                if lap_time >= DEFAULT_MIN_LAP_TIME:
                    self.lap_counter += 1
                    self.lap_durations.append(lap_time)
                    self.get_logger().info(
                        f"Lap {self.lap_counter} completed in {lap_time:.3f}s "
                        f"(crossing s={crossing_s:.2f})"
                    )
                    self.last_lap_time = precise_crossing_time
                else:
                    self.get_logger().warn(f"Ignored lap: too quick ({lap_time:.2f}s)")

        self.previous_position = pos
        self.prev_s = s_cur
        self.prev_timestamp = t

# --- Main launcher ---
def main(args=None):
    rclpy.init(args=args)
    node = RefereeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("[Race Finished]")
        if node.lap_counter > 0:
            node.get_logger().info("Laps recorded.")
        else:
            node.get_logger().info("No complete laps were recorded.")
    finally:
        node.destroy_node()
        rclpy.shutdown()