import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, String

class RefereeNode(Node):
    def __init__(self):
        super().__init__('referee_node')

        # Parameters for start/finish line (two world-frame points)
        self.declare_parameter('start_line_p1', [0.0, 0.0])
        self.declare_parameter('start_line_p2', [10.0, 0.0])
        self.declare_parameter('offtrack_threshold', 0.5)

        p1 = self.get_parameter('start_line_p1').get_parameter_value().double_array_value
        p2 = self.get_parameter('start_line_p2').get_parameter_value().double_array_value
        self.line_p1 = tuple(p1)
        self.line_p2 = tuple(p2)
        self.offtrack_thresh = self.get_parameter('offtrack_threshold').get_parameter_value().double_value

        # State for lap timing
        self.lap_start_time = None
        self.last_side = None
        self.lap_times = []
        self.best_lap = float('inf')

        # State for off-track
        self.is_offtrack = False
        self.offtrack_events = 0

        # Subscribers
        self.create_subscription(Odometry, '/car/odom', self.odom_callback, 10)
        self.create_subscription(Float32, '/planning/error', self.error_callback, 10)

        # Publisher for referee summary
        self.summary_pub = self.create_publisher(String, '/referee/summary', 10)

        self.get_logger().info('Referee node started')

    def odom_callback(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        # Determine which side of start line the robot is on
        side = self._side_of_line((x, y), self.line_p1, self.line_p2)

        # Initialize lap timing
        if self.lap_start_time is None:
            self.lap_start_time = self.get_clock().now().nanoseconds / 1e9
            self.last_side = side
            return

        # Detect crossing from one side to the other
        if side != self.last_side:
            now = self.get_clock().now().nanoseconds / 1e9
            lap_time = now - self.lap_start_time
            self.lap_times.append(lap_time)
            self.best_lap = min(self.best_lap, lap_time)
            self.get_logger().info(f'Lap completed in {lap_time:.2f}s, best lap {self.best_lap:.2f}s')
            self.lap_start_time = now
            self.last_side = side
            self.publish_summary()

    def error_callback(self, msg: Float32):
        err = msg.data
        # Off-track if absolute error exceeds threshold
        if abs(err) > self.offtrack_thresh:
            if not self.is_offtrack:
                self.offtrack_events += 1
                self.is_offtrack = True
                self.get_logger().warn(f'Off-track event #{self.offtrack_events} at time {self.get_clock().now().nanoseconds/1e9:.2f}')
        else:
            self.is_offtrack = False

    def publish_summary(self):
        summary = ('Laps: ' + str(len(self.lap_times)) +
                   ', Best: ' + f'{self.best_lap:.2f}s' +
                   ', Off-track events: ' + str(self.offtrack_events))
        msg = String()
        msg.data = summary
        self.summary_pub.publish(msg)

    @staticmethod
    def _side_of_line(p, a, b):
        # Compute which side of line AB point P lies on (sign of cross product)
        return ((b[0] - a[0])*(p[1] - a[1]) - (b[1] - a[1])*(p[0] - a[0])) > 0


def main(args=None):
    rclpy.init(args=args)
    node = RefereeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Referee node shutting down')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
