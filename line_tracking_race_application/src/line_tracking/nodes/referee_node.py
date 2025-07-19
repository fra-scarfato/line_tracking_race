import os
import csv
from datetime import datetime
import math

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from rclpy.time import Time
from ament_index_python.packages import get_package_share_directory

###STO MODIFICANDO
import json
from std_msgs.msg import String
import time

# Default configuration constants for the referee system
DEFAULT_LINE_POSITION_X = 0.0
DEFAULT_LINE_POSITION_Y = 0.0
DEFAULT_LINE_AXIS = 'y'
DEFAULT_TARGET_YAW = 0.0
DEFAULT_YAW_THRESHOLD = 0.18
DEFAULT_MIN_LAP_TIME = 5.0
DEFAULT_POSITION_THRESHOLD = 0.2
DEFAULT_LINE_LENGTH = 1.0
CSV_OUTPUT_DIR = "race_logs"

class RefereeNode(Node):
    """
    ROS2 node that monitors robot position and detects lap completions.
    Logs detailed lap data to a CSV and a human-readable summary to a Markdown file.
    """
    def __init__(self):
        super().__init__('referee_node')
        self.current_lap_start_time = None
        self._declare_parameters()
        self._get_parameters()
        self._initialize_state()
        self._setup_logging_files() # Renamed for clarity
        self._setup_subscription()
        
        self.stats_pub = self.create_publisher(String, '/referee/stats', 10)
        self.create_timer(0.5, self.publish_stats)
        self.lap_times_pub = self.create_publisher(String, '/referee/lap_times', 10)
        self.create_timer(1.0, self.publish_lap_times)

        
        # Initialize subscriptions, timers, publishers here
        self.get_logger().info(
            f"Referee initialized: "
            f"{'vertical' if self.line_axis=='x' else 'horizontal'} line at "
            f"({self.line_position[0]:.2f}, {self.line_position[1]:.2f}), "
            f"Yaw target: {math.degrees(self.target_yaw):.1f}° ± {math.degrees(self.yaw_threshold):.1f}°, "
            f"Position threshold: {self.position_threshold}m, "
            f"Finish line length: {self.line_length}m"
        )

    def _declare_parameters(self):
        self.declare_parameter('line_position_x', DEFAULT_LINE_POSITION_X)
        self.declare_parameter('line_position_y', DEFAULT_LINE_POSITION_Y)
        self.declare_parameter('line_axis', DEFAULT_LINE_AXIS)
        self.declare_parameter('target_yaw', DEFAULT_TARGET_YAW)
        self.declare_parameter('yaw_threshold_rad', DEFAULT_YAW_THRESHOLD)
        self.declare_parameter('min_lap_time_sec', DEFAULT_MIN_LAP_TIME)
        self.declare_parameter('position_threshold', DEFAULT_POSITION_THRESHOLD)
        self.declare_parameter('line_length', DEFAULT_LINE_LENGTH)

    def _get_parameters(self):
        self.line_position = (self.get_parameter('line_position_x').value, self.get_parameter('line_position_y').value)
        self.line_axis = self.get_parameter('line_axis').value
        self.target_yaw = self.get_parameter('target_yaw').value
        self.yaw_threshold = self.get_parameter('yaw_threshold_rad').value
        self.min_lap_time = self.get_parameter('min_lap_time_sec').value
        self.position_threshold = self.get_parameter('position_threshold').value
        self.line_length = self.get_parameter('line_length').value

    def _initialize_state(self):
        self.previous_position = None
        self.session_start_time = None
        self.last_lap_time = None
        self.lap_counter = 0
        self.lap_durations = []
        self.total_distance = 0.0

    ### STO MODIFICANDO
    def publish_stats(self):
        # Get current ROS time instead of system time
        now = self.get_clock().now().nanoseconds / 1e9
    
        # Use current_lap_start_time if available, otherwise use session_start_time
        if self.current_lap_start_time is not None:
            current_lap_time = now - self.current_lap_start_time
        elif self.session_start_time is not None:
            current_lap_time = now - self.session_start_time
        else:
            current_lap_time = 0.0

        if self.lap_durations:
            best_lap = min(self.lap_durations)
            avg_lap = sum(self.lap_durations) / len(self.lap_durations)
        else:
            best_lap = 0.0
            avg_lap = 0.0

        stats = {
            'lap_counter': self.lap_counter,
            #'current_lap_time': current_lap_time,
            'best_lap_time': best_lap,
            'average_lap_time': avg_lap,
            'total_distance': self.total_distance,
        }
        msg = String()
        msg.data = json.dumps(stats)
        self.stats_pub.publish(msg)


    def publish_lap_times(self):
        msg = String()
        msg.data = json.dumps(self.lap_durations)
        self.lap_times_pub.publish(msg)

    def _setup_logging_files(self):
        # This now sets up both the CSV for data and the Markdown file for the report
        pkg_path = get_package_share_directory("line_tracking_race_application")
        self.log_output_dir = os.path.join(pkg_path, CSV_OUTPUT_DIR)
        os.makedirs(self.log_output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.lap_details_csv = os.path.join(self.log_output_dir, f"lap_details_{timestamp}.csv")
        self.race_summary_file = os.path.join(self.log_output_dir, f"race_summary_{timestamp}.md") # <-- New file

        # Write headers for lap details CSV
        headers = [
            'lap_number','lap_time_seconds','cumulative_time_seconds',
            'crossing_timestamp','yaw_angle_degrees','yaw_deviation_degrees',
            'position_x','position_y','is_best_lap'
        ]
        with open(self.lap_details_csv, 'w', newline='') as f:
            csv.writer(f).writerow(headers)

        self.get_logger().info(f"Logging lap details to: {self.lap_details_csv}")
        self.get_logger().info(f"Race summary will be saved to: {self.race_summary_file}")

    def _setup_subscription(self):
        self.create_subscription(Odometry, '/car/odom', self._on_odometry_received, 10)

    def _on_odometry_received(self, msg: Odometry):
        pos = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        t = Time.from_msg(msg.header.stamp).nanoseconds / 1e9

        if self.session_start_time is None:
            self.session_start_time = t
        if self.previous_position is None:
            self.previous_position = pos
            return

        step = math.hypot(pos[0] - self.previous_position[0], pos[1] - self.previous_position[1])
        self.total_distance += step

        if self._did_cross_line(self.previous_position, pos):
            yaw = self._get_yaw_from_quaternion(msg.pose.pose.orientation)
            diff = abs(self._normalize_angle(yaw - self.target_yaw))
            if diff <= self.yaw_threshold:
                self._process_valid_crossing(t, yaw, pos)

        self.previous_position = pos

    def _did_cross_line(self, start, end):
        cx, cy = self.line_position
        half = self.line_length / 2
        line_p1 = (cx, cy - half) if self.line_axis == 'x' else (cx - half, cy)
        line_p2 = (cx, cy + half) if self.line_axis == 'x' else (cx + half, cy)
        
        # Using a standard line segment intersection algorithm
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        
        return ccw(line_p1, start, end) != ccw(line_p2, start, end) and ccw(line_p1, line_p2, start) != ccw(line_p1, line_p2, end)

    def _process_valid_crossing(self, now, yaw, pos):
        if self.last_lap_time is not None:
            lap_duration = now - self.last_lap_time
            if lap_duration >= self.min_lap_time:
                self.lap_counter += 1
                self.lap_durations.append(lap_duration)
                self._log_lap_to_csv(self.lap_counter, lap_duration, now, yaw, pos)
                self.get_logger().info(f"Lap {self.lap_counter} completed in {lap_duration:.2f}s (Yaw: {math.degrees(yaw):.1f}°)")
                self.current_lap_start_time = now  # <-- Reset timer here
            else:
                self.get_logger().warning(f"Crossing too soon after last lap ({lap_duration:.2f}s). Ignoring.")
        else:
            self.get_logger().info("First finish line crossing detected. Lap timing started.")
            self.current_lap_start_time = now  # <-- Initialize on first crossing
        self.last_lap_time = now

    def _log_lap_to_csv(self, lap_num, lap_time, timestamp, yaw, pos):
        is_best_lap = lap_time == min(self.lap_durations)
        row = [
            lap_num,
            round(lap_time, 3),
            round(sum(self.lap_durations), 3),
            datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            round(math.degrees(yaw), 2),
            round(math.degrees(abs(self._normalize_angle(yaw - self.target_yaw))), 2),
            round(pos[0], 3),
            round(pos[1], 3),
            is_best_lap
        ]
        with open(self.lap_details_csv, 'a', newline='') as f:
            csv.writer(f).writerow(row)

    def _generate_race_summary_report(self):
        """Generates a human-readable race summary in a Markdown file."""
        if not self.lap_durations:
            self.get_logger().warn("No laps were completed. Cannot generate summary report.")
            return

        # --- Calculate Statistics ---
        total_laps = len(self.lap_durations)
        total_time = sum(self.lap_durations)
        best_lap = min(self.lap_durations)
        worst_lap = max(self.lap_durations)
        avg_lap = total_time / total_laps
        var = sum((t - avg_lap) ** 2 for t in self.lap_durations) / total_laps
        std_dev = math.sqrt(var)
        consistency = max(0, (1 - std_dev / avg_lap) * 100)
        
        trend = 0.0
        if total_laps >= 2:
            mid = total_laps // 2
            trend = (sum(self.lap_durations[:mid]) / mid) - (sum(self.lap_durations[mid:]) / (total_laps - mid))

        # --- Build Markdown Content ---
        report = []
        report.append("# Race Summary\n")
        report.append(f"- **Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append("---\n")
        report.append("## 🏁 Overall Performance\n")
        report.append("| Metric                | Value      |")
        report.append("| --------------------- | ---------- |")
        report.append(f"| **Total Laps** | **{total_laps}** |")
        report.append(f"| **Total Racing Time** | **{total_time:.2f} s** |")
        report.append(f"| **Best Lap Time** | **{best_lap:.2f} s** |")
        report.append(f"| **Average Lap Time** | **{avg_lap:.2f} s** |\n")
        report.append("---\n")
        report.append("## 📊 Key Statistics\n")
        report.append(f"- **Worst Lap Time:** {worst_lap:.2f} s")
        report.append(f"- **Lap Time Std. Deviation:** {std_dev:.2f} s")
        report.append(f"- **Consistency Score:** {consistency:.2f} %")
        trend_str = f"{abs(trend):.2f} s ({'Faster' if trend > 0 else 'Slower'})"
        report.append(f"- **Improvement Trend:** {trend_str}")
        report.append(f"- **Total Distance Traveled:** {self.total_distance:.2f} m")
        report.append("---\n")
        report.append("## 🏎️ Lap-by-Lap Analysis\n")
        report.append("| Lap | Time (s) | Cumulative (s) | Δ to Best (s) | Δ to Avg (s) |")
        report.append("| :-- | :------- | :------------- | :------------ | :----------- |")
        
        cumulative_time = 0
        for i, t in enumerate(self.lap_durations, 1):
            cumulative_time += t
            delta_best = t - best_lap
            delta_avg = t - avg_lap
            lap_str = f"**{t:.2f}**" if t == best_lap else f"{t:.2f}"
            best_str = f"**{delta_best:+.2f}**" if t == best_lap else f"{delta_best:+.2f}"
            avg_str = f"**{delta_avg:+.2f}**" if t == best_lap else f"{delta_avg:+.2f}"
            report.append(f"| {i:<3} | {lap_str:<8} | {cumulative_time:<14.2f} | {best_str:<13} | {avg_str:<12} |")
        
        report.append("\n---\n")
        report.append("## ⚙️ Race Configuration\n")
        line_desc = 'Horizontal' if self.line_axis == 'y' else 'Vertical'
        report.append(f"- **Finish Line:** {line_desc} line at ({self.line_position[0]:.2f}, {self.line_position[1]:.2f})")
        report.append(f"- **Finish Line Length:** {self.line_length} m")
        report.append(f"- **Target Yaw:** {math.degrees(self.target_yaw):.1f}° (± {math.degrees(self.yaw_threshold):.1f}°)")
        report.append(f"- **Position Threshold:** {self.position_threshold} m")
        report.append(f"- **Minimum Lap Time:** {self.min_lap_time} s")

        # --- Write to File ---
        try:
            with open(self.race_summary_file, 'w') as f:
                f.write("\n".join(report))
            self.get_logger().info(f"Human-readable summary report saved to: {self.race_summary_file}")
        except IOError as e:
            self.get_logger().error(f"Failed to write summary report: {e}")

    @staticmethod
    def _quaternion_to_yaw(o):
        return math.atan2(2.0 * (o.w * o.z + o.x * o.y), 1.0 - 2.0 * (o.y * o.y + o.z * o.z))

    def _get_yaw_from_quaternion(self, o):
        return self._normalize_angle(self._quaternion_to_yaw(o))

    @staticmethod
    def _normalize_angle(a):
        return (a + math.pi) % (2 * math.pi) - math.pi

def main(args=None):
    rclpy.init(args=args)
    node = RefereeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("\n[Race Finished]")
        if node.lap_counter > 0:
            best = min(node.lap_durations)
            avg = sum(node.lap_durations) / len(node.lap_durations)
            node.get_logger().info(
                f"Summary: {node.lap_counter} laps completed | "
                f"Best: {best:.2f}s | Avg: {avg:.2f}s | "
                f"Dist: {node.total_distance:.2f}m"
            )
            node._generate_race_summary_report() # Generate the new .md report
        else:
            node.get_logger().info("No complete laps were recorded.")
    finally:
        node.destroy_node()

if __name__ == '__main__':
    main()