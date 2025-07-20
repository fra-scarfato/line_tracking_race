import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.ticker import MaxNLocator
import numpy as np
from line_tracking.track_utility import TrackUtility
import math

class RefereeGuiNode(Node):
    def __init__(self):
        super().__init__('referee_gui_node')

        # Initialize stats
        self.stats = {
            'lap_counter': 0,
            'current_velocity': 0.0,
            'last_lap_time': 0.0,
            'best_lap_time': 0.0,
            'average_lap_time': 0.0,
            'total_distance': 0.0,
        }
        self.lap_times = []
        self.last_lap_count = 0  # Track last lap count to control redraw
        self.track_utility = TrackUtility()
        
        # Car position tracking with filtering
        self.car_position = {'x': 0.0, 'y': 0.0}
        self.previous_car_position = {'x': 0.0, 'y': 0.0}
        self.car_trail = {'x': [], 'y': []}  # Store car trail
        self.max_trail_length = 50  # Maximum number of trail points to keep
        
        # Position filtering parameters
        self.max_position_jump = 0.9  # Maximum allowed position jump in meters
        self.position_initialized = False
        self.outlier_count = 0
        self.max_outliers_before_reset = 3
        
        # Track visualization state
        self.track_initialized = False
        self.canvas_ready = False
        
        # Debug variables
        self.debug_counter = 0
        self.last_car_pos_logged = None

        # Setup tkinter with increased width for track visualization
        self.root = tk.Tk()
        self.root.title("Race Dashboard")

        # Configure root background and padding
        self.root.configure(bg="#1a1a1a")  # Dark background
        self.root.geometry("1400x700")
        
        # Create main container
        main_frame = tk.Frame(self.root, bg="#1a1a1a")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left side - Stats and charts
        left_frame = tk.Frame(main_frame, bg="#1a1a1a")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 10))
        
        # Right side - Track simulation
        right_frame = tk.Frame(main_frame, bg="#2a2a2a", relief=tk.RAISED, bd=2)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Title label
        title_label = tk.Label(left_frame, text="Race Dashboard", font=('Arial', 20, 'bold'),
                               bg="#1a1a1a", fg="#00ff00")
        title_label.pack(pady=(0, 15))

        # Stats frame
        stats_frame = tk.Frame(left_frame, bg="#333333", relief=tk.RAISED, bd=1)
        stats_frame.pack(fill=tk.X, pady=(0, 15))

        self.labels = {}
        for i, key in enumerate(self.stats.keys()):
            row_frame = tk.Frame(stats_frame, bg="#333333")
            row_frame.pack(fill=tk.X, padx=10, pady=2)
            
            label_title = tk.Label(row_frame, text=key.replace('_', ' ').title() + ":",
                                   font=('Arial', 12), bg="#333333", fg="#ffffff", anchor='w')
            label_title.pack(side=tk.LEFT)
            
            self.labels[key] = tk.Label(row_frame, text="0", font=('Arial', 12, 'bold'),
                                        bg="#333333", fg="#00ff00", anchor='e')
            self.labels[key].pack(side=tk.RIGHT)

        # Setup matplotlib for charts only (left side)
        self.fig, (self.ax_times, self.ax_deltas) = plt.subplots(2, 1, figsize=(6, 6))
        self.fig.patch.set_facecolor("#1a1a1a")
        
        # Set consistent subplot spacing
        self.fig.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.1, hspace=0.4)
        
        for ax in [self.ax_times, self.ax_deltas]:
            ax.set_facecolor("#2a2a2a")

        self.canvas = FigureCanvasTkAgg(self.fig, master=left_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Setup track simulation canvas (right side)
        self.setup_track_canvas(right_frame)

        # Draw initial empty plots
        self.draw_initial_plots()

        # Subscribers
        self.subscription = self.create_subscription(
            String,
            '/referee/stats',
            self.stats_callback,
            10
        )
        self.lap_times_subscription = self.create_subscription(
            String,
            '/referee/lap_times',
            self.lap_times_callback,
            10
        )
        
        # Add odom subscription for velocity and position
        from nav_msgs.msg import Odometry
        self.odom_subscription = self.create_subscription(
            Odometry,
            '/car/odom',
            self.odom_callback,
            10
        )

        # Periodic updates
        self.update_gui()
        self.ros_spin_once()

        # Handle clean shutdown on window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def filter_position(self, new_x, new_y):
        """
        Filter incoming position data to remove outliers and sudden jumps.
        Returns filtered position or None if position should be rejected.
        """
        if not self.position_initialized:
            # First position - always accept
            self.position_initialized = True
            self.previous_car_position = {'x': new_x, 'y': new_y}
            self.outlier_count = 0
            return new_x, new_y
        
        # Calculate distance from previous position
        dx = new_x - self.previous_car_position['x']
        dy = new_y - self.previous_car_position['y']
        distance = math.sqrt(dx*dx + dy*dy)
        
        # Check if this is a reasonable position jump
        if distance > self.max_position_jump:
            self.outlier_count += 1
            self.get_logger().warn(
                f"Large position jump detected: {distance:.2f}m "
                f"from ({self.previous_car_position['x']:.2f}, {self.previous_car_position['y']:.2f}) "
                f"to ({new_x:.2f}, {new_y:.2f}). Outlier count: {self.outlier_count}"
            )
            
            # If we have too many outliers in a row, accept the new position
            if self.outlier_count >= self.max_outliers_before_reset:
                self.get_logger().info("Too many outliers - accepting new position as reset")
                self.previous_car_position = {'x': new_x, 'y': new_y}
                self.outlier_count = 0
                return new_x, new_y
            
            # Reject this position update
            return None, None
        
        # Position seems reasonable
        self.outlier_count = 0
        self.previous_car_position = {'x': new_x, 'y': new_y}
        return new_x, new_y

    def is_position_on_track(self, x, y):
        """
        Check if a position is reasonably close to the track.
        Returns True if position seems valid for the track.
        """
        if not self.track_initialized or len(self.track_points) == 0:
            return True  # Can't validate without track data
        
        # Find minimum distance to any track point
        min_distance = float('inf')
        for track_point in self.track_points:
            dx = x - track_point[0]
            dy = y - track_point[1]
            distance = math.sqrt(dx*dx + dy*dy)
            min_distance = min(min_distance, distance)
        
        # Allow some tolerance around the track (e.g., 2 meters)
        track_tolerance = 2.0
        return min_distance <= track_tolerance

    def setup_track_canvas(self, parent):
        """Setup the track simulation canvas."""
        # Title for track section
        track_title = tk.Label(parent, text="Live Track View", font=('Arial', 16, 'bold'),
                              bg="#2a2a2a", fg="#00ff00")
        track_title.pack(pady=10)
        
        # Create canvas for track simulation
        self.track_canvas = tk.Canvas(parent, bg="#000000", highlightthickness=0)
        self.track_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Track and car elements
        self.track_points = []
        self.car_dot = None
        self.trail_line = None
        self.track_scale = 0.0
        self.track_offset_x = 0
        self.track_offset_y = 0
        
        # Bind resize event
        self.track_canvas.bind('<Configure>', self.on_canvas_resize)
        
        # Initialize track after a delay to ensure canvas is ready
        self.root.after(500, self.initialize_track_delayed)

    def initialize_track_delayed(self):
        """Initialize track with multiple attempts if needed."""
        self.get_logger().info("Attempting to initialize track...")
        try:
            # Check if canvas has proper dimensions
            canvas_width = self.track_canvas.winfo_width()
            canvas_height = self.track_canvas.winfo_height()
            
            if canvas_width <= 1 or canvas_height <= 1:
                self.get_logger().info("Canvas not ready, retrying in 200ms...")
                self.root.after(200, self.initialize_track_delayed)
                return
            
            self.canvas_ready = True
            self.get_logger().info(f"Canvas ready: {canvas_width}x{canvas_height}")
            
            # Try to get track points
            track_points = self.track_utility.get_world_points()
            
            if track_points is not None and len(track_points) > 0:
                self.track_points = np.array(track_points)
                self.get_logger().info(f"Got {len(self.track_points)} track points")
                
                # Log track bounds for debugging
                xs = self.track_points[:,0]
                ys = self.track_points[:,1]
                self.min_x, self.max_x = xs.min(), xs.max()
                self.min_y, self.max_y = ys.min(), ys.max()
                self.get_logger().info(f"Track bounds: X=[{self.min_x:.2f}, {self.max_x:.2f}], Y=[{self.min_y:.2f}, {self.max_y:.2f}]")
                
                self.setup_track_scaling()
                self.draw_track()
                self.create_car_elements()
                self.track_initialized = True
                self.get_logger().info("Track initialized successfully!")
            else:
                self.get_logger().warn("No track data available, showing placeholder")
                self.show_placeholder()
                # Retry after some time in case track data becomes available
                self.root.after(2000, self.initialize_track_delayed)
                
        except Exception as e:
            self.get_logger().error(f"Error initializing track: {e}")
            self.show_placeholder()
            # Retry after some time
            self.root.after(2000, self.initialize_track_delayed)

    def show_placeholder(self):
        """Show placeholder text on canvas."""
        self.track_canvas.delete("all")
        canvas_width = self.track_canvas.winfo_width()
        canvas_height = self.track_canvas.winfo_height()
        self.track_canvas.create_text(
            canvas_width//2, canvas_height//2,
            text="Loading track data...\nWaiting for TrackUtility",
            fill="#ffffff", 
            font=('Arial', 14),
            justify=tk.CENTER
        )

    def setup_track_scaling(self):
        """Calculate scale & margins so the track fits and Y is flipped."""
        if len(self.track_points) == 0:
            return

        w = self.track_canvas.winfo_width()
        h = self.track_canvas.winfo_height()

        # Track bounds in world coords
        xs = self.track_points[:,0]
        ys = self.track_points[:,1]
        self.min_x, self.max_x = xs.min(), xs.max()
        self.min_y, self.max_y = ys.min(), ys.max()
        track_w = self.max_x - self.min_x
        track_h = self.max_y - self.min_y

        # Compute uniform scale to fit (with 50px padding)
        pad = 50
        avail_w = w - 2*pad
        avail_h = h - 2*pad
        scale = min(avail_w/track_w, avail_h/track_h) if track_w > 0 and track_h > 0 else 1.0

        # Margins on each side so it's centered
        margin_x = (w - track_w*scale) / 2
        margin_y = (h - track_h*scale) / 2

        # Save for world→canvas
        self.track_scale = scale
        self.margin_x = margin_x
        self.margin_y = margin_y
        
        # Log scaling info for debugging
        self.get_logger().info(f"Scaling setup: scale={scale:.3f}, margins=({margin_x:.1f}, {margin_y:.1f})")
        self.get_logger().info(f"Canvas size: {w}x{h}, Track size: {track_w:.2f}x{track_h:.2f}")

    def world_to_canvas(self, wx, wy):
        """
        Convert world coordinates to canvas coordinates.
        This version handles proper coordinate system transformation.
        """
        # Transform X: left to right (normal)
        cx = self.margin_x + (wx - self.min_x) * self.track_scale
        
        # Transform Y: flip so that positive Y in world goes up on screen
        # Canvas Y=0 is at top, so we need to flip
        cy = self.margin_y + (self.max_y - wy) * self.track_scale
        
        return cx, cy

    def canvas_to_world(self, cx, cy):
        """Convert canvas coordinates back to world coordinates (for debugging)."""
        wx = (cx - self.margin_x) / self.track_scale + self.min_x
        wy = self.max_y - (cy - self.margin_y) / self.track_scale
        return wx, wy

    def draw_track(self):
        """Draw the track on canvas with proper offset."""
        if len(self.track_points) == 0:
            return
            
        TRACK_WIDTH = 0.2  # Adjust this based on your actual track width
        
        # Clear existing track elements
        self.track_canvas.delete("track")
        
        # Calculate offset points (outward normal)
        offset_points = []
        n = len(self.track_points)
        
        for i in range(n):
            prev = self.track_points[(i-1)%n]
            curr = self.track_points[i]
            next_p = self.track_points[(i+1)%n]
            
            # Calculate normal vector
            v1 = curr - prev
            v2 = next_p - curr
            normal = np.array([-v1[1], v1[0]])  # Perpendicular vector
            if np.linalg.norm(normal) > 0:  # Avoid division by zero
                normal = normal / np.linalg.norm(normal)  # Normalize
            
            # Offset point
            offset_point = curr + normal * TRACK_WIDTH/2
            offset_points.append(offset_point)
        
        # Convert offset points to canvas coordinates
        canvas_points = []
        for point in offset_points:
            x, y = self.world_to_canvas(point[0], point[1])
            canvas_points.extend([x, y])
        
        # Close the loop
        if len(canvas_points) >= 4:
            canvas_points.extend([canvas_points[0], canvas_points[1]])
            self.track_canvas.create_line(
                canvas_points,
                fill="#F2F2F2",
                width=5,
                smooth=True,
                tags="track"
            )

    def create_car_elements(self):
        """Create car dot and trail elements."""
        # Create car dot at current position or origin
        car_x, car_y = self.car_position['x'], self.car_position['y']
        
        # If car position is still at origin, try to place it at the first track point
        if car_x == 0.0 and car_y == 0.0 and len(self.track_points) > 0:
            car_x, car_y = self.track_points[0][0], self.track_points[0][1]
        
        start_canvas_x, start_canvas_y = self.world_to_canvas(car_x, car_y)
        
        self.car_dot = self.track_canvas.create_oval(
            start_canvas_x - 8, start_canvas_y - 8, start_canvas_x + 8, start_canvas_y + 8,
            fill="#ff0000",
            outline="#ffffff",
            width=2,
            tags="car"
        )
        
        # Create trail line (initially empty)
        self.trail_line = self.track_canvas.create_line(
            start_canvas_x, start_canvas_y, start_canvas_x, start_canvas_y,
            fill="#ff6666",
            width=2,
            smooth=True,
            tags="trail"
        )
        
        self.get_logger().info(f"Car elements created at world({car_x:.2f}, {car_y:.2f}) -> canvas({start_canvas_x:.1f}, {start_canvas_y:.1f})")

    def update_car_visualization(self):
        """Update car position and trail on canvas."""
        if not self.track_initialized or self.car_dot is None:
            return
            
        # Convert car position to canvas coordinates
        car_canvas_x, car_canvas_y = self.world_to_canvas(
            self.car_position['x'], 
            self.car_position['y']
        )
        
        # Check if canvas coordinates are reasonable
        canvas_width = self.track_canvas.winfo_width()
        canvas_height = self.track_canvas.winfo_height()
        
        if (car_canvas_x < -100 or car_canvas_x > canvas_width + 100 or
            car_canvas_y < -100 or car_canvas_y > canvas_height + 100):
            self.get_logger().warn(
                f"Car canvas position out of bounds: ({car_canvas_x:.1f}, {car_canvas_y:.1f}) "
                f"Canvas size: {canvas_width}x{canvas_height}"
            )
        
        # Debug logging every 100 updates
        self.debug_counter += 1
        if self.debug_counter % 100 == 0:
            current_pos = (self.car_position['x'], self.car_position['y'])
            if current_pos != self.last_car_pos_logged:
                self.get_logger().info(
                    f"Car position: world({self.car_position['x']:.3f}, {self.car_position['y']:.3f}) "
                    f"-> canvas({car_canvas_x:.1f}, {car_canvas_y:.1f})"
                )
                self.last_car_pos_logged = current_pos
                
                # Check if car is within track bounds
                if hasattr(self, 'min_x') and hasattr(self, 'max_x'):
                    if (self.min_x <= self.car_position['x'] <= self.max_x and 
                        self.min_y <= self.car_position['y'] <= self.max_y):
                        self.get_logger().info("Car is within track bounds")
                    else:
                        self.get_logger().warn(f"Car is OUTSIDE track bounds! Track bounds: X=[{self.min_x:.2f}, {self.max_x:.2f}], Y=[{self.min_y:.2f}, {self.max_y:.2f}]")
        
        # Update car dot position
        dot_size = 8
        self.track_canvas.coords(
            self.car_dot,
            car_canvas_x - dot_size, car_canvas_y - dot_size,
            car_canvas_x + dot_size, car_canvas_y + dot_size
        )
        
        # Add current position to trail (limit trail length)
        self.car_trail['x'].append(self.car_position['x'])
        self.car_trail['y'].append(self.car_position['y'])

    def on_canvas_resize(self, event):
        """Handle canvas resize event."""
        self.canvas_width  = event.width
        self.canvas_height = event.height
        
        if self.track_initialized and len(self.track_points) > 0:
            self.get_logger().info(f"Canvas resized to {event.width}x{event.height}")
            self.setup_track_scaling()
            self.draw_track()
            # Recreate car elements with new scaling
            if self.car_dot:
                self.track_canvas.delete("car")
                self.track_canvas.delete("trail")
                self.create_car_elements()

    def draw_initial_plots(self):
        for ax, title, ylabel in [
            (self.ax_times, "Lap Times", "Time (s)"),
            (self.ax_deltas, "Lap Time Delta", "Delta (s)")
        ]:
            ax.clear()
            ax.set_facecolor("#2a2a2a")
            ax.set_title(title, fontsize=14, color='#ffffff')
            ax.set_xlabel("Lap Number", fontsize=10, color='#cccccc')
            ax.set_ylabel(ylabel, fontsize=10, color='#cccccc')
            ax.grid(True, linestyle='--', alpha=0.3, color='#666666')
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.tick_params(axis='both', labelsize=10, colors='#cccccc')
            
            # Style the spines
            for spine in ax.spines.values():
                spine.set_color('#666666')
                spine.set_linewidth(1)
            
            if title == "Lap Time Delta":
                ax.axhline(0, color='#666666', linestyle='--', linewidth=1)
        
        self.canvas.draw()

    def stats_callback(self, msg):
        try:
            data = json.loads(msg.data)
            for key in self.stats:
                if key in data:
                    self.stats[key] = data[key]
        except json.JSONDecodeError:
            self.get_logger().error("Failed to decode JSON from /referee/stats")

    def ros_spin_once(self):
        rclpy.spin_once(self, timeout_sec=0)
        self.root.after(10, self.ros_spin_once)

    def update_gui(self):
        # Update last_lap_time from lap_times list
        if self.lap_times:
            self.stats['last_lap_time'] = self.lap_times[-1]
        else:
            self.stats['last_lap_time'] = 0.0

        # Update labels with color coding
        self.labels['lap_counter'].config(text=str(self.stats['lap_counter']))
        self.labels['last_lap_time'].config(text=f"{self.stats['last_lap_time']:.2f} s")
        self.labels['best_lap_time'].config(text=f"{self.stats['best_lap_time']:.2f} s")
        self.labels['average_lap_time'].config(text=f"{self.stats['average_lap_time']:.2f} s")
        self.labels['total_distance'].config(text=f"{self.stats['total_distance']:.1f} m")
        
        # Color code velocity
        velocity = self.stats['current_velocity']
        if velocity > 50:
            vel_color = "#ff0000"  # Red for high speed
        elif velocity > 30:
            vel_color = "#ffff00"  # Yellow for medium speed
        else:
            vel_color = "#00ff00"  # Green for low speed
        
        self.labels['current_velocity'].config(text=f"{velocity:.1f} km/h", fg=vel_color)

        # Update car visualization
        self.update_car_visualization()

        # Update plots only if new lap has been added
        if len(self.lap_times) != self.last_lap_count:
            self.last_lap_count = len(self.lap_times)

            # Update Lap Times Plot
            self.ax_times.clear()
            self.ax_times.set_title("Lap Times", fontsize=14, color='#ffffff')
            self.ax_times.set_xlabel("Lap Number", fontsize=10, color='#cccccc')
            self.ax_times.set_ylabel("Time (s)", fontsize=10, color='#cccccc')
            self.ax_times.grid(True, linestyle='--', alpha=0.3, color='#666666')
            self.ax_times.xaxis.set_major_locator(MaxNLocator(integer=True))
            self.ax_times.tick_params(axis='both', labelsize=10, colors='#cccccc')
            self.ax_times.set_facecolor("#2a2a2a")
            
            for spine in self.ax_times.spines.values():
                spine.set_color('#666666')
                spine.set_linewidth(1)

            if self.lap_times:
                laps = list(range(1, len(self.lap_times) + 1))
                self.ax_times.plot(
                    laps,
                    self.lap_times,
                    marker='o',
                    color='#00ff00',
                    linestyle='-',
                    linewidth=2,
                    markersize=6,
                    label='Lap Times'
                )
                avg = self.stats['average_lap_time']
                if avg > 0:
                    self.ax_times.axhline(
                        avg,
                        color='#ff6600',
                        linestyle='--',
                        linewidth=2,
                        label="Average"
                    )
                legend = self.ax_times.legend(fontsize=9, facecolor='#2a2a2a', 
                                            edgecolor='#666666', labelcolor='#ffffff')

            # Update Lap Time Delta Plot
            self.ax_deltas.clear()
            self.ax_deltas.set_title("Lap Time Delta", fontsize=14, color='#ffffff')
            self.ax_deltas.set_xlabel("Lap Number", fontsize=10, color='#cccccc')
            self.ax_deltas.set_ylabel("Delta (s)", fontsize=10, color='#cccccc')
            self.ax_deltas.grid(True, linestyle='--', alpha=0.3, color='#666666')
            self.ax_deltas.tick_params(axis='both', labelsize=10, colors='#cccccc')
            self.ax_deltas.xaxis.set_major_locator(MaxNLocator(integer=True))
            self.ax_deltas.set_facecolor("#2a2a2a")
            self.ax_deltas.axhline(0, color='#666666', linestyle='--', linewidth=1)
            
            for spine in self.ax_deltas.spines.values():
                spine.set_color('#666666')
                spine.set_linewidth(1)

            if len(self.lap_times) > 1:
                deltas = [self.lap_times[i] - self.lap_times[i - 1] for i in range(1, len(self.lap_times))]
                laps = list(range(2, len(self.lap_times) + 1))
                colors = ['#00ff00' if d < 0 else '#ff0000' for d in deltas]
                self.ax_deltas.bar(laps, deltas, color=colors, alpha=0.8, edgecolor='#666666')
                
                max_delta = max(abs(d) for d in deltas) if deltas else 1
                self.ax_deltas.set_ylim(-max_delta * 1.2, max_delta * 1.2)
            else:
                self.ax_deltas.set_ylim(-1, 1)

            self.canvas.draw()

        self.root.after(50, self.update_gui)  # 50ms updates for smooth car movement

    def lap_times_callback(self, msg):
        try:
            self.lap_times = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().error("Failed to decode JSON from /referee/lap_times")

    def odom_callback(self, msg):
        # Calculate linear velocity
        linear_x = msg.twist.twist.linear.x
        linear_y = msg.twist.twist.linear.y
        velocity_ms = (linear_x**2 + linear_y**2)**0.5
        velocity_kmh = velocity_ms * 3.6
        self.stats['current_velocity'] = velocity_kmh
        
        # Get new position from odometry
        new_x = msg.pose.pose.position.x
        new_y = msg.pose.pose.position.y
        
        # Apply position filtering to remove outliers
        filtered_x, filtered_y = self.filter_position(new_x, new_y)
        
        # Only update position if it passed the filter
        if filtered_x is not None and filtered_y is not None:
            # Additional validation: check if position is reasonable relative to track
            if self.is_position_on_track(filtered_x, filtered_y):
                self.car_position['x'] = filtered_x
                self.car_position['y'] = filtered_y
            else:
                self.get_logger().warn(
                    f"Position ({filtered_x:.2f}, {filtered_y:.2f}) is too far from track, ignoring"
                )
        else:
            self.get_logger().debug("Position update filtered out as outlier")

    def on_close(self):
        self.destroy_node()
        rclpy.shutdown()
        self.root.destroy()


def main(args=None):
    rclpy.init(args=args)
    node = RefereeGuiNode()

    try:
        node.root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()