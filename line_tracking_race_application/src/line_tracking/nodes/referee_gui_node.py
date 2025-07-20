import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.ticker import MaxNLocator
import numpy as np
import math

from line_tracking.track_utility import TrackUtility


class RefereeGuiNode(Node):
    """
    ROS2 node with a Tkinter GUI displaying live race statistics and track view.
    Features:
        - Real-time lap stats and velocity
        - Matplotlib charts for lap times and deltas
        - Canvas rendering of track and car trail
        - Position filtering to reject outliers
    """
    def __init__(self):
        super().__init__('referee_gui_node')

        # Initialize data structures
        self.stats = {  # Published stats
            'lap_counter': 0,
            'current_velocity': 0.0,
            'last_lap_time': 0.0,
            'best_lap_time': 0.0,
            'average_lap_time': 0.0,
            'total_distance': 0.0,
        }
        self.lap_times = []            # Raw lap times for plotting
        self.last_lap_count = 0        # To detect when to redraw plots

        # Car position and trail for canvas simulation
        self.car_position = {'x': 0.0, 'y': 0.0}
        self.previous_car_position = {'x': 0.0, 'y': 0.0}
        self.car_trail = {'x': [], 'y': []}
        self.max_trail_length = 50

        # Outlier filtering parameters
        self.position_initialized = False
        self.max_position_jump = 5.0       # meters
        self.outlier_count = 0
        self.max_outliers_before_reset = 3

        # Track rendering state
        self.track_utility = TrackUtility()
        self.track_points = []             # Filled once canvas is ready
        self.track_initialized = False
        self.canvas_ready = False

        # Build GUI layout
        self._init_tkinter()
        self._init_charts()
        self.setup_track_canvas(self.right_frame)
        self.draw_initial_plots()

        # ROS subscriptions
        self.subscription = self.create_subscription(
            String, '/referee/stats', self.stats_callback, 10)
        self.lap_times_subscription = self.create_subscription(
            String, '/referee/lap_times', self.lap_times_callback, 10)
        from nav_msgs.msg import Odometry
        self.odom_subscription = self.create_subscription(
            Odometry, '/car/odom', self.odom_callback, 10)

        # Start update loops
        self.update_gui()
        self.ros_spin_once()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _init_tkinter(self):
        """Initialize Tkinter root, frames, and stat labels."""
        self.root = tk.Tk()
        self.root.title("Race Dashboard")
        self.root.configure(bg="#1a1a1a")
        self.root.geometry("1400x700")

        main_frame = tk.Frame(self.root, bg="#1a1a1a")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left: stats & charts
        self.left_frame = tk.Frame(main_frame, bg="#1a1a1a")
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0,10))
        # Right: track view
        self.right_frame = tk.Frame(main_frame, bg="#2a2a2a", relief=tk.RAISED, bd=2)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        tk.Label(self.left_frame, text="Race Dashboard", font=('Arial',20,'bold'),
                 bg="#1a1a1a", fg="#00ff00").pack(pady=(0,15))

        # Stats panel
        stats_panel = tk.Frame(self.left_frame, bg="#333333", relief=tk.RAISED, bd=1)
        stats_panel.pack(fill=tk.X, pady=(0,15))
        self.labels = {}
        for key in self.stats:
            frame = tk.Frame(stats_panel, bg="#333333")
            frame.pack(fill=tk.X, padx=10, pady=2)
            tk.Label(frame, text=f"{key.replace('_',' ').title()}:", font=('Arial',12),
                     bg="#333333", fg="#ffffff").pack(side=tk.LEFT)
            self.labels[key] = tk.Label(frame, text="0", font=('Arial',12,'bold'),
                                         bg="#333333", fg="#00ff00")
            self.labels[key].pack(side=tk.RIGHT)

    def _init_charts(self):
        """Initialize Matplotlib Figure and Axes for lap time plots."""
        self.fig, (self.ax_times, self.ax_deltas) = plt.subplots(2,1,figsize=(6,6))
        self.fig.patch.set_facecolor("#1a1a1a")
        self.fig.subplots_adjust(left=0.15,right=0.95,top=0.95,bottom=0.1,hspace=0.4)
        for ax in (self.ax_times, self.ax_deltas):
            ax.set_facecolor("#2a2a2a")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.left_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def draw_initial_plots(self):
        """Prepare empty lap times and delta plots with styling."""
        for ax, title, ylabel in [
            (self.ax_times, "Lap Times", "Time (s)"),
            (self.ax_deltas, "Lap Time Delta", "Delta (s)")
        ]:
            self._style_axis(ax, title, ylabel)
            if ax is self.ax_deltas:
                ax.axhline(0, linestyle='--', linewidth=1)
        self.canvas.draw()

    def _style_axis(self, ax, title, ylabel):
        """Apply consistent styling to a Matplotlib Axes."""
        ax.clear()
        ax.set_title(title, fontsize=14, color='#ffffff')
        ax.set_xlabel("Lap Number", fontsize=10, color='#cccccc')
        ax.set_ylabel(ylabel, fontsize=10, color='#cccccc')
        ax.grid(True, linestyle='--', alpha=0.3, color='#666666')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.tick_params(axis='both', labelsize=10, colors='#cccccc')
        for spine in ax.spines.values():
            spine.set_color('#666666'); spine.set_linewidth(1)

    # -- ROS Callbacks ------------------------------------------------
    def stats_callback(self, msg):
        try:
            data = json.loads(msg.data)
            for k,v in data.items():
                if k in self.stats:
                    self.stats[k] = v
        except json.JSONDecodeError:
            self.get_logger().error("Invalid JSON in stats")

    def lap_times_callback(self, msg):
        try:
            self.lap_times = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().error("Invalid JSON in lap_times")

    def odom_callback(self, msg):
        # Compute speed (m/s to km/h)
        vx, vy = msg.twist.twist.linear.x, msg.twist.twist.linear.y
        self.stats['current_velocity'] = math.hypot(vx,vy)*3.6
        # Filter position
        new_x, new_y = msg.pose.pose.position.x, msg.pose.pose.position.y
        fx, fy = self.filter_position(new_x, new_y)
        if fx is not None and self._is_on_track(fx,fy):
            self.car_position.update({'x':fx,'y':fy})
        else:
            self.get_logger().warn("Rejected outlier or off-track pos")

    # -- Position Filtering --------------------------------------------
    def filter_position(self, nx, ny):
        if not self.position_initialized:
            self.position_initialized = True
            self.previous_car_position.update({'x':nx,'y':ny})
            return nx, ny
        dx, dy = nx - self.previous_car_position['x'], ny - self.previous_car_position['y']
        dist = math.hypot(dx, dy)
        if dist > self.max_position_jump:
            self.outlier_count += 1
            if self.outlier_count >= self.max_outliers_before_reset:
                self.get_logger().info("Resetting after outliers")
                self.outlier_count = 0
                self.previous_car_position.update({'x':nx,'y':ny})
                return nx, ny
            return None, None
        self.outlier_count = 0
        self.previous_car_position.update({'x':nx,'y':ny})
        return nx, ny

    def _is_on_track(self, x, y):
        """Check if (x,y) lies within tolerance of track points."""
        if not self.track_initialized:
            return True
        dists = np.hypot(self.track_points[:,0]-x,
                         self.track_points[:,1]-y)
        return dists.min() <= 2.0

    # -- Track Canvas Setup and Drawing --------------------------------
    def setup_track_canvas(self, parent):
        tk.Label(parent, text="Live Track View", font=('Arial',16,'bold'),
                 bg="#2a2a2a", fg="#00ff00").pack(pady=10)
        self.track_canvas = tk.Canvas(parent, bg="#000000", highlightthickness=0)
        self.track_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.track_canvas.bind('<Configure>', self.on_canvas_resize)
        self.root.after(500, self._delayed_track_init)

    def _delayed_track_init(self):
        """Retry until canvas is ready, then load track points and draw."""
        w, h = self.track_canvas.winfo_width(), self.track_canvas.winfo_height()
        if w<2 or h<2:
            self.root.after(200, self._delayed_track_init); return
        self.canvas_ready = True
        pts = self.track_utility.get_world_points()
        if pts:
            self.track_points = np.array(pts)
            self._compute_scaling()
            self.draw_track()
            self._create_car_dot()
            self.track_initialized = True
        else:
            self.show_placeholder(); self.root.after(2000, self._delayed_track_init)

    def show_placeholder(self):
        self.track_canvas.delete("all")
        w,h = self.track_canvas.winfo_width(), self.track_canvas.winfo_height()
        self.track_canvas.create_text(w//2, h//2,
            text="Loading track...",
            fill="#ffffff", font=('Arial',14), justify=tk.CENTER)

    def _compute_scaling(self):
        xs, ys = self.track_points[:,0], self.track_points[:,1]
        self.min_x, self.max_x, self.min_y, self.max_y = xs.min(), xs.max(), ys.min(), ys.max()
        w,h = self.track_canvas.winfo_width(), self.track_canvas.winfo_height()
        pad=50; avail_w, avail_h = w-2*pad, h-2*pad
        scale = min(avail_w/(self.max_x-self.min_x), avail_h/(self.max_y-self.min_y))
        self.track_scale, self.margin_x, self.margin_y = scale, (w-scale*(self.max_x-self.min_x))/2, (h-scale*(self.max_y-self.min_y))/2
        self.get_logger().info(f"Scale={scale:.3f}, margins=({self.margin_x:.1f},{self.margin_y:.1f})")

    def world_to_canvas(self, wx, wy):
        cx = self.margin_x + (wx-self.min_x)*self.track_scale
        cy = self.margin_y + (self.max_y-wy)*self.track_scale
        return cx, cy

    def draw_track(self):
        self.track_canvas.delete("track")
        pts = self.track_points
        # Compute offset polyline for track width
        offsets=[]
        for i in range(len(pts)):
            prev, cur, nxt = pts[i-1], pts[i], pts[(i+1)%len(pts)]
            tangent = cur-prev
            normal = np.array([-tangent[1], tangent[0]]);
            if np.linalg.norm(normal)>0: normal/=np.linalg.norm(normal)
            offsets.append(cur + normal*0.1)
        coords = []
        for p in offsets+offsets[:1]: coords.extend(self.world_to_canvas(*p))
        self.track_canvas.create_line(coords, fill="#F2F2F2", width=5, smooth=True, tags="track")

    def _create_car_dot(self):
        # Fix: Properly handle the case when car_position['y'] is 0.0
        if self.car_position['y'] == 0.0 and len(self.track_points) > 0:
            # Use the first track point coordinates
            x, y = self.track_points[0][0], self.track_points[0][1]
        else:
            x, y = self.car_position['x'], self.car_position['y']
        
        cx, cy = self.world_to_canvas(x, y)
        r = 8
        self.car_dot = self.track_canvas.create_oval(cx-r,cy-r,cx+r,cy+r, fill="#ff0000", outline="#ffffff", width=2)
        self.trail_line = self.track_canvas.create_line(cx,cy,cx,cy, fill="#ff6666", width=2, smooth=True, tags="trail")

    def update_car_visualization(self):
        if not self.track_initialized: return
        cx,cy = self.world_to_canvas(self.car_position['x'], self.car_position['y'])
        self.track_canvas.coords(self.car_dot, cx-8,cy-8,cx+8,cy+8)
        # Append to trail and redraw line
        self.car_trail['x'].append(self.car_position['x'])
        self.car_trail['y'].append(self.car_position['y'])

    # -- Main update loops ---------------------------------------------
    def ros_spin_once(self):
        rclpy.spin_once(self, timeout_sec=0)
        self.root.after(10, self.ros_spin_once)

    def update_gui(self):
        # Update stats labels
        self.labels['lap_counter'].config(text=str(self.stats['lap_counter']))
        self.labels['last_lap_time'].config(text=f"{self.lap_times[-1] if self.lap_times else 0:.2f} s")
        self.labels['best_lap_time'].config(text=f"{self.stats['best_lap_time']:.2f} s")
        self.labels['average_lap_time'].config(text=f"{self.stats['average_lap_time']:.2f} s")
        self.labels['total_distance'].config(text=f"{self.stats['total_distance']:.1f} m")
        vel=self.stats['current_velocity'];
        color="#00ff00" if vel<=30 else ("#ffff00" if vel<=50 else "#ff0000")
        self.labels['current_velocity'].config(text=f"{vel:.1f} km/h", fg=color)

        # Redraw track and car trail
        self.update_car_visualization()

        # Redraw plots if new lap
        if len(self.lap_times)!=self.last_lap_count:
            self.last_lap_count=len(self.lap_times)
            self._style_axis(self.ax_times, "Lap Times", "Time (s)")
            if self.lap_times:
                laps=list(range(1,len(self.lap_times)+1))
                self.ax_times.plot(laps, self.lap_times, marker='o', linestyle='-', linewidth=2)
                avg=self.stats['average_lap_time']
                if avg>0: self.ax_times.axhline(avg, linestyle='--', linewidth=2)
            self._style_axis(self.ax_deltas, "Lap Time Delta", "Delta (s)")
            if len(self.lap_times)>1:
                deltas=[j-i for i,j in zip(self.lap_times,self.lap_times[1:])]
                laps=list(range(2,len(self.lap_times)+1))
                self.ax_deltas.bar(laps,deltas)
            self.canvas.draw()

        self.root.after(50, self.update_gui)

    def on_canvas_resize(self, event):
        if self.track_initialized:
            self._compute_scaling(); self.draw_track(); self.update_car_visualization()

    def on_close(self):
        self.destroy_node(); rclpy.shutdown(); self.root.destroy()



def main(args=None):
    rclpy.init(args=args)
    node = RefereeGuiNode()
    try:
        node.root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node(); rclpy.shutdown()


if __name__ == '__main__':
    main()