import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.ticker import MaxNLocator


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

        # Setup tkinter
        self.root = tk.Tk()
        self.root.title("Race Stats")

        # Configure root background and padding
        self.root.configure(bg="#e6f0ff")  # Light blue background
        self.root.geometry("700x650")
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

        # Title label with padding
        title_label = tk.Label(self.root, text="Race Statistics", font=('Arial', 24, 'bold'),
                               bg="#e6f0ff", fg="#003366")
        title_label.grid(row=0, column=0, columnspan=2, pady=(20, 15), sticky='n')

        # Stats frame for centering labels
        stats_frame = tk.Frame(self.root, bg="#cce0ff")
        stats_frame.grid(row=1, column=0, columnspan=2, sticky='ew', padx=50, pady=(0, 15))
        stats_frame.grid_columnconfigure(0, weight=1)
        stats_frame.grid_columnconfigure(1, weight=1)

        self.labels = {}
        for i, key in enumerate(self.stats.keys()):
            label_title = tk.Label(stats_frame, text=key.replace('_', ' ').title() + ":",
                                   font=('Arial', 16), bg="#cce0ff", fg="#003366", anchor='e')
            label_title.grid(row=i, column=0, sticky='e', padx=(0, 10), pady=5)
            self.labels[key] = tk.Label(stats_frame, text="0", font=('Arial', 16),
                                        bg="#cce0ff", fg="#003366", anchor='w')
            self.labels[key].grid(row=i, column=1, sticky='w', padx=(10, 0), pady=5)

        # Setup matplotlib: 2 subplots for lap times and lap deltas
        self.fig, (self.ax_times, self.ax_deltas) = plt.subplots(2, 1, figsize=(7, 6))
        self.fig.patch.set_facecolor("#e6f0ff")  # Match tkinter background
        
        # Set consistent subplot spacing from the beginning
        self.fig.subplots_adjust(left=0.12, right=0.95, top=0.93, bottom=0.12, hspace=0.4)
        
        self.ax_times.set_facecolor("#f0f8ff")  # Very light blue for plot background
        self.ax_deltas.set_facecolor("#f0f8ff")  # Very light blue for plot background

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().grid(row=2, column=0, columnspan=2, sticky='nsew', padx=20, pady=15)

        # Configure grid weights for proper resizing
        self.root.grid_rowconfigure(2, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

        # Draw initial empty plots with titles, axes, and grids visible
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
        
        # Add odom subscription for velocity
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

    def draw_initial_plots(self):
        for ax, title, ylabel in [
            (self.ax_times, "Lap Times", "Time (s)"),
            (self.ax_deltas, "Lap Time Delta", "Delta (s)")
        ]:
            ax.clear()
            ax.set_facecolor("#f0f8ff")  # Very light blue background
            ax.set_title(title, fontsize=16, color='#003366')  # Dark blue title
            ax.set_xlabel("Lap Number", fontsize=10, color='#003366', labelpad=2)  # Smaller dark blue labels with less padding
            ax.set_ylabel(ylabel, fontsize=10, color='#003366')  # Smaller dark blue labels
            ax.grid(True, linestyle='--', alpha=0.7, color='#6699cc')  # Medium blue grid
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.tick_params(axis='both', labelsize=12, colors='#003366')  # Dark blue tick labels
            
            # Style the spines (plot borders)
            for spine in ax.spines.values():
                spine.set_color('#003366')
                spine.set_linewidth(1.2)
            
            if title == "Lap Time Delta":
                ax.axhline(0, color='#003366', linestyle='--', linewidth=1.5)
        
        # Apply consistent spacing and draw
        self.fig.subplots_adjust(left=0.12, right=0.95, top=0.93, bottom=0.12, hspace=0.4)
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

        # Update labels
        self.labels['lap_counter'].config(text=str(self.stats['lap_counter']))
        self.labels['last_lap_time'].config(text=f"{self.stats['last_lap_time']:.2f} s")
        self.labels['best_lap_time'].config(text=f"{self.stats['best_lap_time']:.2f} s")
        self.labels['average_lap_time'].config(text=f"{self.stats['average_lap_time']:.2f} s")
        self.labels['total_distance'].config(text=f"{self.stats['total_distance']:.2f} m")
        self.labels['current_velocity'].config(text=f"{self.stats['current_velocity']:.2f} km/h")

        # Update plots only if new lap has been added
        if len(self.lap_times) != self.last_lap_count:
            self.last_lap_count = len(self.lap_times)

            # Update Lap Times Plot
            self.ax_times.clear()
            self.ax_times.set_title("Lap Times", fontsize=16, color='#003366')
            self.ax_times.set_xlabel("Lap Number", fontsize=10, color='#003366', labelpad=2)
            self.ax_times.set_ylabel("Time (s)", fontsize=10, color='#003366')
            self.ax_times.grid(True, linestyle='--', alpha=0.7, color='#6699cc')
            self.ax_times.xaxis.set_major_locator(MaxNLocator(integer=True))
            self.ax_times.tick_params(axis='both', labelsize=12, colors='#003366')
            self.ax_times.set_facecolor("#f0f8ff")
            
            # Style the spines
            for spine in self.ax_times.spines.values():
                spine.set_color('#003366')
                spine.set_linewidth(1.2)

            if self.lap_times:
                laps = list(range(1, len(self.lap_times) + 1))
                self.ax_times.plot(
                    laps,
                    self.lap_times,
                    marker='o',
                    color='#0066cc',  # Medium blue for data line
                    linestyle='-',
                    label='Lap Times'
                )
                avg = self.stats['average_lap_time']
                self.ax_times.axhline(
                    avg,
                    color='#cc3300',  # Red for average line
                    linestyle='--',
                    label="Average Lap Time"
                )
                legend = self.ax_times.legend(fontsize=10)
                legend.get_frame().set_facecolor('#e6f0ff')  # Light blue legend background
                for text in legend.get_texts():
                    text.set_color('#003366')

            # Update Lap Time Delta Plot
            self.ax_deltas.clear()
            self.ax_deltas.set_title("Lap Time Delta", fontsize=16, color='#003366')
            self.ax_deltas.set_xlabel("Lap Number", fontsize=10, color='#003366', labelpad=2)
            self.ax_deltas.set_ylabel("Delta (s)", fontsize=10, color='#003366')
            self.ax_deltas.grid(True, linestyle='--', alpha=0.7, color='#6699cc')
            self.ax_deltas.tick_params(axis='both', labelsize=12, colors='#003366')
            self.ax_deltas.xaxis.set_major_locator(MaxNLocator(integer=True))
            self.ax_deltas.set_facecolor("#f0f8ff")
            self.ax_deltas.axhline(0, color='#003366', linestyle='--', linewidth=1.5)
            
            # Style the spines
            for spine in self.ax_deltas.spines.values():
                spine.set_color('#003366')
                spine.set_linewidth(1.2)

            if len(self.lap_times) > 1:
                deltas = [self.lap_times[i] - self.lap_times[i - 1] for i in range(1, len(self.lap_times))]
                laps = list(range(2, len(self.lap_times) + 1))
                colors = ['#009900' if d < 0 else '#cc3300' for d in deltas]  # Green/Red with better contrast
                self.ax_deltas.bar(laps, deltas, color=colors, edgecolor='#003366')
                
                # Create custom legend with green and red entries
                from matplotlib.patches import Patch
                legend_elements = [Patch(facecolor='#009900', label='Delta Negative'),
                                   Patch(facecolor='#cc3300', label='Delta Positive')]
                legend = self.ax_deltas.legend(handles=legend_elements, fontsize=10)
                legend.get_frame().set_facecolor('#e6f0ff')  # Light blue legend background
                for text in legend.get_texts():
                    text.set_color('#003366')

                # Auto-scale y-axis to symmetrical range for clarity
                max_delta = max(abs(d) for d in deltas)
                self.ax_deltas.set_ylim(-max_delta * 1.2, max_delta * 1.2)
            else:
                self.ax_deltas.set_ylim(-1, 1)  # Default range if no data

            # Apply consistent spacing instead of tight_layout
            self.fig.subplots_adjust(left=0.12, right=0.95, top=0.93, bottom=0.12, hspace=0.4)
            self.canvas.draw()

        self.root.after(1000, self.update_gui)

    def lap_times_callback(self, msg):
        try:
            self.lap_times = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().error("Failed to decode JSON from /referee/lap_times")

    def odom_callback(self, msg):
        # Calculate linear velocity magnitude from x and y components
        linear_x = msg.twist.twist.linear.x
        linear_y = msg.twist.twist.linear.y
        velocity_ms = (linear_x**2 + linear_y**2)**0.5
        velocity_kmh = velocity_ms * 3.6
        self.stats['current_velocity'] = velocity_kmh

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