import tkinter as tk
from tkinter import ttk, messagebox
import threading
import queue
import time
import random
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List
from datetime import datetime
import logging
import json
import os
import uuid

# Configure logging to output information-level messages
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ThreadState(Enum):
    """Enum representing the possible states of a simulated thread."""
    NEW = "New"
    RUNNABLE = "Runnable"
    RUNNING = "Running"
    WAITING = "Waiting"
    TERMINATED = "Terminated"

class ThreadModel(Enum):
    """Enum representing threading models for simulation."""
    MANY_TO_ONE = "Many-to-One"
    ONE_TO_MANY = "One-to-Many"
    MANY_TO_MANY = "Many-to-Many"

@dataclass
class ThreadInfo:
    """Data class to store information about a simulated thread."""
    tid: int                    # Thread ID
    state: ThreadState         # Current state of the thread
    model: ThreadModel         # Threading model
    progress: int = 0          # Progress percentage (0-100)
    cpu_time: float = 0.0      # Total CPU time used
    start_time: float = 0.0    # Time when thread started
    wait_time: float = 0.0     # Total time spent waiting
    task_count: int = 0        # Number of tasks completed

    def __post_init__(self):
        """Ensure default attributes for backward compatibility."""
        if not hasattr(self, 'wait_time'):
            self.wait_time = 0.0
        if not hasattr(self, 'task_count'):
            self.task_count = 0

class Semaphore:
    """A semaphore for thread synchronization with a specified initial count."""
    def __init__(self, count: int):
        self.count = count                     # Initial semaphore count
        self.lock = threading.Lock()           # Lock for thread-safe operations
        self.condition = threading.Condition(self.lock)  # Condition for waiting

    def wait(self):
        """Decrement the semaphore count, block if count is zero."""
        with self.condition:
            while self.count <= 0:
                self.condition.wait()          # Wait until count is positive
            self.count -= 1                    # Decrement count

    def signal(self):
        """Increment the semaphore count and notify a waiting thread."""
        with self.condition:
            self.count += 1                    # Increment count
            self.condition.notify()            # Wake up one waiting thread

class Monitor:
    """A reentrant monitor for thread synchronization."""
    def __init__(self):
        self.lock = threading.Lock()           # Lock for monitor
        self.condition = threading.Condition(self.lock)  # Condition for waiting
        self.owner: int | None = None          # Current owner thread ID
        self.entry_count = 0                   # Reentrant entry count

    def enter(self, thread_id: int):
        """Enter the monitor, block if owned by another thread."""
        with self.condition:
            while self.owner is not None and self.owner != thread_id:
                self.condition.wait()          # Wait if monitor is owned
            self.owner = thread_id             # Set current thread as owner
            self.entry_count += 1              # Increment reentrant count

    def exit(self, thread_id: int):
        """Exit the monitor, release if no longer needed."""
        with self.condition:
            if self.owner != thread_id:
                raise ValueError(f"Thread {thread_id} does not own monitor")
            self.entry_count -= 1              # Decrement reentrant count
            if self.entry_count == 0:
                self.owner = None              # Release monitor if count is 0
                self.condition.notify_all()    # Notify all waiting threads

class ThreadSimulator:
    """A GUI-based thread synchronization simulator."""
    def __init__(self, root: tk.Tk):
        # Initialize core attributes
        self.root = root                       # Tkinter root window
        self.root.title("ThreadSync Pro - Enhanced Simulator")
        self.root.minsize(800, 600)            # Set minimum window size
        self.threads: List[threading.Thread] = []  # List of active threads
        self.thread_info: Dict[int, ThreadInfo] = {}  # Thread ID to ThreadInfo mapping
        self.thread_map: Dict[threading.Thread, int] = {}  # Thread object to Thread ID
        self.running = False                   # Simulation running state
        self.model = ThreadModel.MANY_TO_MANY  # Default threading model
        self.semaphore = Semaphore(2)          # Semaphore with initial count 2
        self.monitor = Monitor()               # Monitor for synchronization
        self.task_queue = queue.Queue()        # Queue for UI update tasks
        self.speed = 1.0                       # Simulation speed multiplier
        self.scheduler_thread: threading.Thread | None = None  # Scheduler thread
        self.paused = False                    # Pause state
        self.pause_event = threading.Event()   # Event to control pause/resume
        self.pause_event.set()                 # Initially not paused
        self.canvas_width = 600                # Canvas width for visualization
        self.canvas_height = 400               # Canvas height for visualization
        self.thread_lock = threading.Lock()     # Lock for thread list modifications
        self.setup_ui()                        # Set up the GUI
        self.setup_styles()                    # Configure widget styles
        self.setup_logging()                   # Set up logging panel

    def setup_styles(self):
        """Configure Tkinter widget styles for consistent appearance."""
        self.style = ttk.Style()               # Create style object
        self.style.theme_use('clam')           # Use 'clam' theme
        self.style.configure("TButton", padding=6, font=('Arial', 10))
        self.style.configure("TLabel", padding=3, font=('Arial', 10))
        self.style.configure("TLabelframe.Label", font=('Arial', 10, 'bold'))
        self.style.configure("Blue.TButton", background="#2196F3", foreground="white")
        self.style.configure("Red.TButton", background="#F44336", foreground="white")

    def setup_logging(self):
        """Set up the logging panel in the UI."""
        self.log_frame = ttk.LabelFrame(self.main_frame, text="Event Log", padding="8")
        self.log_frame.grid(row=4, column=0, sticky="nsew", pady=5)
        self.log_text = tk.Text(self.log_frame, height=5, width=50, state='disabled')
        self.log_text.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(self.log_frame, orient="vertical", command=self.log_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log_text['yscrollcommand'] = scrollbar.set
        self.log_frame.columnconfigure(0, weight=1)  # Allow log text to expand

    def setup_ui(self):
        """Set up the main UI components."""
        # Main frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Control panel
        control_frame = ttk.LabelFrame(self.main_frame, text="Control Panel", padding="8")
        control_frame.grid(row=0, column=0, sticky="ew", pady=5)

        # Thread count input
        ttk.Label(control_frame, text="Threads:").grid(row=0, column=0, padx=5)
        self.thread_count_var = tk.StringVar(value="4")
        ttk.Spinbox(control_frame, from_=1, to=20, textvariable=self.thread_count_var, 
                   width=5).grid(row=0, column=1, padx=5)

        # Thread model selection
        ttk.Label(control_frame, text="Model:").grid(row=0, column=2, padx=5)
        self.model_var = tk.StringVar(value=ThreadModel.MANY_TO_MANY.value)
        model_combo = ttk.Combobox(control_frame, textvariable=self.model_var, 
                                 state="readonly", width=12)
        model_combo['values'] = [m.value for m in ThreadModel]
        model_combo.grid(row=0, column=3, padx=5)

        # Speed control
        ttk.Label(control_frame, text="Speed:").grid(row=0, column=4, padx=5)
        self.speed_var = tk.DoubleVar(value=1.0)
        speed_scale = ttk.Scale(control_frame, from_=0.1, to=3.0, variable=self.speed_var, 
                              orient="horizontal", length=100)
        speed_scale.grid(row=0, column=5, padx=5)
        speed_scale.bind("<ButtonRelease-1>", lambda _: self.update_speed())

        # Control buttons
        ttk.Button(control_frame, text="Start", command=self.start_simulation, 
                  style="Blue.TButton").grid(row=0, column=6, padx=5)
        self.pause_button = ttk.Button(control_frame, text="Pause", command=self.toggle_pause)
        self.pause_button.grid(row=0, column=7, padx=5)
        ttk.Button(control_frame, text="Stop", command=self.stop_simulation, 
                  style="Red.TButton").grid(row=0, column=8, padx=5)
        ttk.Button(control_frame, text="Reset", command=self.reset_simulation).grid(row=0, column=9, padx=5)
        ttk.Button(control_frame, text="Save", command=self.save_simulation).grid(row=0, column=10, padx=5)
        ttk.Button(control_frame, text="Load", command=self.load_simulation).grid(row=0, column=11, padx=5)
        ttk.Button(control_frame, text="Export Logs", command=self.export_logs).grid(row=0, column=12, padx=5)

        # Visualization panel
        self.vis_frame = ttk.LabelFrame(self.main_frame, text="Thread Visualization", padding="8")
        self.vis_frame.grid(row=1, column=0, sticky="nsew", pady=5)
        self.canvas = tk.Canvas(self.vis_frame, bg="#f5f5f5", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", self.on_canvas_resize)
        self.vis_frame.columnconfigure(0, weight=1)
        self.vis_frame.rowconfigure(0, weight=1)

        # Status panel
        status_frame = ttk.LabelFrame(self.main_frame, text="System Status", padding="8")
        status_frame.grid(row=2, column=0, sticky="ew", pady=5)
        self.semaphore_label = ttk.Label(status_frame, text="Semaphore: 2")
        self.semaphore_label.grid(row=0, column=0, padx=10)
        self.monitor_label = ttk.Label(status_frame, text="Monitor: None")
        self.monitor_label.grid(row=0, column=1, padx=10)
        self.cpu_label = ttk.Label(status_frame, text="CPU Time: 0.0s")
        self.cpu_label.grid(row=0, column=2, padx=10)
        self.active_label = ttk.Label(status_frame, text="Active Threads: 0")
        self.active_label.grid(row=0, column=3, padx=10)

        # Metrics panel
        self.metrics_frame = ttk.LabelFrame(self.main_frame, text="Performance Metrics", padding="8")
        self.metrics_frame.grid(row=3, column=0, sticky="ew", pady=5)
        self.wait_time_label = ttk.Label(self.metrics_frame, text="Avg Wait Time: 0.0s")
        self.wait_time_label.grid(row=0, column=0, padx=10)
        self.throughput_label = ttk.Label(self.metrics_frame, text="Throughput: 0.0 tasks/s")
        self.throughput_label.grid(row=0, column=1, padx=10)

        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.rowconfigure(1, weight=1)

    def on_canvas_resize(self, event):
        """Update canvas dimensions when resized."""
        self.canvas_width = event.width
        self.canvas_height = event.height

    def update_speed(self):
        """Update simulation speed based on scale value."""
        self.speed = self.speed_var.get()

    def log_event(self, message: str):
        """Log an event to the UI, ensuring thread safety."""
        def update_log():
            self.log_text.configure(state='normal')
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
            self.log_text.see(tk.END)
            self.log_text.configure(state='disabled')
        self.root.after(0, update_log)

    def save_simulation(self):
        """Save the current simulation state to a JSON file."""
        if not self.thread_info:
            messagebox.showwarning("Warning", "No simulation to save!")
            return
        state = {
            "thread_info": {tid: {
                "tid": info.tid,
                "state": info.state.value,
                "model": info.model.value,
                "progress": info.progress,
                "cpu_time": info.cpu_time,
                "start_time": info.start_time,
                "wait_time": info.wait_time,
                "task_count": info.task_count
            } for tid, info in self.thread_info.items()},
            "semaphore_count": self.semaphore.count,
            "model": self.model.value,
            "speed": self.speed
        }
        try:
            with open("simulation.json", "w") as f:
                json.dump(state, f, indent=2)
            self.log_event("Simulation saved to simulation.json")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            messagebox.showerror("Error", f"Failed to save simulation: {e}")

    def load_simulation(self):
        """Load a simulation state from a JSON file."""
        if not os.path.exists("simulation.json"):
            messagebox.showerror("Error", "No saved simulation found!")
            return
        try:
            with open("simulation.json", "r") as f:
                state = json.load(f)
            if not all(k in state for k in ["thread_info", "semaphore_count", "model", "speed"]):
                raise ValueError("Invalid simulation file format")
            self.reset_simulation()  # Clear current simulation
            self.model = ThreadModel(state["model"])
            self.model_var.set(self.model.value)
            self.speed = state["speed"]
            self.speed_var.set(self.speed)
            self.semaphore = Semaphore(state["semaphore_count"])
            self.semaphore_label.config(text=f"Semaphore: {self.semaphore.count}")
            for tid, info in state["thread_info"].items():
                t_info = ThreadInfo(
                    tid=int(tid),
                    state=ThreadState(info["state"]),
                    model=ThreadModel(info["model"]),
                    progress=info["progress"],
                    cpu_time=info["cpu_time"],
                    start_time=info["start_time"],
                    wait_time=info["wait_time"],
                    task_count=info["task_count"]
                )
                self.thread_info[int(tid)] = t_info
                if t_info.state != ThreadState.TERMINATED:
                    t = threading.Thread(target=self.thread_function, args=(int(tid),))
                    t.daemon = True
                    with self.thread_lock:
                        self.threads.append(t)
                        self.thread_map[t] = int(tid)
                    t.start()
            self.running = True
            self.scheduler_thread = threading.Thread(target=self.scheduler)
            self.scheduler_thread.daemon = True
            self.scheduler_thread.start()
            self.update_ui()
            self.log_event("Simulation loaded from simulation.json")
        except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
            messagebox.showerror("Error", f"Failed to load simulation: {e}")

    def export_logs(self):
        """Export the event log to a text file."""
        try:
            self.log_text.configure(state='normal')
            logs = self.log_text.get(1.0, tk.END)
            self.log_text.configure(state='disabled')
            with open("simulation_logs.txt", "w") as f:
                f.write(logs)
            self.log_event("Logs exported to simulation_logs.txt")
        except FileNotFoundError as e:
            messagebox.showerror("Error", f"Failed to export logs: {e}")

    def start_simulation(self):
        """Start a new simulation with user-specified parameters."""
        if self.running:
            messagebox.showwarning("Warning", "Simulation already running!")
            return
        try:
            thread_count = int(self.thread_count_var.get())
            if thread_count < 1 or thread_count > 20:
                raise ValueError("Thread count must be between 1 and 20")
            if self.model_var.get() not in [m.value for m in ThreadModel]:
                raise ValueError("Invalid thread model selected")
        except ValueError as e:
            messagebox.showerror("Error", str(e))
            return

        self.running = True
        self.paused = False
        self.pause_event.set()
        self.model = ThreadModel(self.model_var.get())
        with self.thread_lock:
            self.threads = []
            self.thread_info = {}
            self.thread_map = {}
        self.canvas.delete("all")
        self.log_event("Starting simulation")

        for i in range(thread_count):
            t_info = ThreadInfo(tid=i, state=ThreadState.NEW, model=self.model, 
                              start_time=time.time())
            self.thread_info[i] = t_info
            t = threading.Thread(target=self.thread_function, args=(i,))
            t.daemon = True
            with self.thread_lock:
                self.threads.append(t)
                self.thread_map[t] = i
            t.start()
            self.log_event(f"Thread {i} created")

        self.scheduler_thread = threading.Thread(target=self.scheduler)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()
        self.update_ui()

    def toggle_pause(self):
        """Toggle the pause state of the simulation."""
        if self.paused:
            self.pause_event.set()
            self.paused = False
            self.pause_button.config(text="Pause")
            self.log_event("Simulation resumed")
        else:
            self.pause_event.clear()
            self.paused = True
            self.pause_button.config(text="Resume")
            self.log_event("Simulation paused")

    def stop_simulation(self):
        """Stop the simulation and clean up."""
        self.running = False
        self.paused = False
        self.pause_event.set()
        self.pause_button.config(text="Pause")
        self.log_event("Simulation stopped")

    def reset_simulation(self):
        """Reset the simulation to its initial state."""
        self.stop_simulation()
        self.canvas.delete("all")
        self.semaphore = Semaphore(2)
        self.semaphore_label.config(text="Semaphore: 2")
        self.monitor_label.config(text="Monitor: None")
        self.cpu_label.config(text="CPU Time: 0.0s")
        self.active_label.config(text="Active Threads: 0")
        self.wait_time_label.config(text="Avg Wait Time: 0.0s")
        self.throughput_label.config(text="Throughput: 0.0 tasks/s")
        with self.thread_lock:
            self.thread_info = {}
            self.threads = []
            self.thread_map = {}
        self.log_text.configure(state='normal')
        self.log_text.delete(1.0, tk.END)
        self.log_text.configure(state='disabled')
        self.log_event("Simulation reset")

    def thread_function(self, tid: int):
        """Simulate a thread's execution with synchronization."""
        t_info = self.thread_info[tid]
        t_info.state = ThreadState.RUNNABLE
        def safe_queue(task): self.root.after(0, lambda: self.task_queue.put(task))
        safe_queue((tid, "start"))
        wait_time = 0.0
        task_count = 0

        while self.running and t_info.progress < 100:
            self.pause_event.wait()  # Respect pause state
            start_time = time.time()
            if t_info.state == ThreadState.RUNNING:
                self.speed = self.speed_var.get()
                try:
                    if t_info.model == ThreadModel.MANY_TO_ONE:
                        # Simple execution without synchronization
                        time.sleep(random.uniform(0.1, 0.3) / self.speed)
                        t_info.progress += random.randint(2, 10)
                        task_count += 1
                        safe_queue((tid, "progress"))
                    elif t_info.model == ThreadModel.ONE_TO_MANY:
                        # Use semaphore for synchronization
                        if random.random() < 0.3:
                            self.semaphore.wait()
                            t_info.state = ThreadState.WAITING
                            safe_queue((tid, "semaphore_acquire"))
                            time.sleep(random.uniform(0.1, 0.5) / self.speed)
                            t_info.progress += random.randint(10, 25)
                            task_count += 1
                            self.semaphore.signal()
                            safe_queue((tid, "semaphore_release"))
                        else:
                            time.sleep(0.05 / self.speed)
                    else:  # Many-to-Many
                        # Use both semaphore and monitor
                        if random.choice([True, False]):
                            self.semaphore.wait()
                            t_info.state = ThreadState.WAITING
                            safe_queue((tid, "semaphore_acquire"))
                            time.sleep(random.uniform(0.2, 0.8) / self.speed)
                            t_info.progress += random.randint(5, 20)
                            task_count += 1
                            self.semaphore.signal()
                            safe_queue((tid, "semaphore_release"))
                        else:
                            self.monitor.enter(tid)
                            t_info.state = ThreadState.WAITING
                            safe_queue((tid, "monitor_enter"))
                            time.sleep(random.uniform(0.2, 0.8) / self.speed)
                            t_info.progress += random.randint(5, 20)
                            task_count += 1
                            self.monitor.exit(tid)
                            safe_queue((tid, "monitor_exit"))

                    t_info.state = ThreadState.RUNNABLE
                    t_info.cpu_time = time.time() - t_info.start_time
                    safe_queue((tid, "progress"))
                except Exception as e:
                    logger.error(f"Thread {tid} error: {e}")
                    self.log_event(f"Thread {tid} error: {e}")
            wait_time += time.time() - start_time
            elapsed = time.time() - start_time
            sleep_time = max(0, (0.02 / self.speed) - elapsed)
            time.sleep(sleep_time)

        if self.running:
            t_info.state = ThreadState.TERMINATED
            t_info.wait_time = wait_time
            t_info.task_count = task_count
            safe_queue((tid, "terminated"))
            self.log_event(f"Thread {tid} terminated")

    def scheduler(self):
        """Schedule threads for execution in a round-robin fashion."""
        current_thread = 0
        while self.running:
            self.pause_event.wait()
            with self.thread_lock:
                if self.threads and current_thread < len(self.threads):
                    t_info = self.thread_info[self.thread_map[self.threads[current_thread]]]
                    if t_info.state == ThreadState.RUNNABLE:
                        t_info.state = ThreadState.RUNNING
                        self.task_queue.put((self.thread_map[self.threads[current_thread]], "running"))
                        time.sleep(0.2 / self.speed)
                        if t_info.state == ThreadState.RUNNING:
                            t_info.state = ThreadState.RUNNABLE
                            self.task_queue.put((self.thread_map[self.threads[current_thread]], "runnable"))
                    current_thread = (current_thread + 1) % len(self.threads)
            time.sleep(0.01 / self.speed)

    def update_ui(self):
        """Update the UI with thread states and performance metrics."""
        if not self.running:
            return

        # Process queued tasks (limit to 100 to prevent overload)
        try:
            for _ in range(100):
                tid, action = self.task_queue.get_nowait()
                t_info = self.thread_info.get(tid)
                if not t_info:
                    continue
                if action == "semaphore_acquire":
                    self.semaphore_label.config(text=f"Semaphore: {self.semaphore.count}")
                    self.log_event(f"Thread {tid} acquired semaphore")
                elif action == "semaphore_release":
                    self.semaphore_label.config(text=f"Semaphore: {self.semaphore.count}")
                    self.log_event(f"Thread {tid} released semaphore")
                elif action == "monitor_enter":
                    self.monitor_label.config(text=f"Monitor: Thread {tid}")
                    self.log_event(f"Thread {tid} entered monitor")
                elif action == "monitor_exit":
                    self.monitor_label.config(text="Monitor: None")
                    self.log_event(f"Thread {tid} exited monitor")
                elif action == "progress":
                    self.cpu_label.config(text=f"CPU Time: {sum(t.cpu_time for t in self.thread_info.values()):.1f}s")
                    self.active_label.config(text=f"Active Threads: {sum(1 for t in self.thread_info.values() if t.state != ThreadState.TERMINATED)}")
        except queue.Empty:
            pass

        # Update performance metrics
        if self.thread_info:
            total_wait = sum(t.wait_time for t in self.thread_info.values())
            avg_wait = total_wait / len(self.thread_info) if total_wait > 0 else 0
            total_tasks = sum(t.task_count for t in self.thread_info.values())
            runtime = max((time.time() - t.start_time) for t in self.thread_info.values()) if self.thread_info else 1
            throughput = total_tasks / runtime if runtime > 0 else 0
            self.wait_time_label.config(text=f"Avg Wait Time: {avg_wait:.1f}s")
            self.throughput_label.config(text=f"Throughput: {throughput:.1f} tasks/s")

        # Update visualization on canvas
        self.canvas.delete("all")
        thread_count = len(self.thread_info)
        bar_height = min(30, max(20, (self.canvas_height - 20 * thread_count) // max(thread_count, 1)))

        state_colors = {
            ThreadState.NEW: "#B0BEC5",
            ThreadState.RUNNABLE: "#2196F3",
            ThreadState.RUNNING: "#4CAF50",
            ThreadState.WAITING: "#FF9800",
            ThreadState.TERMINATED: "#F44336"
        }

        for i, t_info in self.thread_info.items():
            y = i * (bar_height + 20) + 20
            progress_width = (self.canvas_width - 120) * min(t_info.progress, 100) / 100
            
            # Draw background rectangle
            self.canvas.create_rectangle(60, y, self.canvas_width - 60, y + bar_height, 
                                      fill="#e0e0e0", outline="")
            
            # Draw progress bar
            self.canvas.create_rectangle(60, y, 60 + progress_width, y + bar_height,
                                      fill=state_colors[t_info.state], outline="")
            
            # Draw border
            self.canvas.create_rectangle(60, y, self.canvas_width - 60, y + bar_height, 
                                      outline="#757575", width=1)
            
            # Draw thread ID and status text
            self.canvas.create_text(40, y + bar_height // 2, 
                                  text=f"T{i}", font=('Arial', 10, 'bold'), anchor="e")
            status_text = f"{t_info.state.value} ({t_info.progress}%) - {t_info.cpu_time:.1f}s"
            self.canvas.create_text(self.canvas_width - 40, y + bar_height // 2,
                                  text=status_text, font=('Arial', 9), anchor="e")
            
            # Draw semaphore/monitor indicators
            if t_info.state == ThreadState.WAITING or t_info.state == ThreadState.RUNNING:
                if self.semaphore.count < 2:
                    self.canvas.create_oval(20, y + bar_height // 4, 30, y + 3 * bar_height // 4,
                                          fill="#FF5722", outline="")
                if self.monitor.owner == t_info.tid:
                    self.canvas.create_rectangle(10, y + bar_height // 4, 20, y + 3 * bar_height // 4,
                                              fill="#4CAF50", outline="")

        # Clean up terminated threads
        terminated = [tid for tid, t_info in self.thread_info.items() if t_info.state == ThreadState.TERMINATED]
        for tid in terminated:
            thread = next((t for t, t_id in self.thread_map.items() if t_id == tid), None)
            if thread and not thread.is_alive():
                self.log_event(f"Cleaning up terminated thread {tid}")
                with self.thread_lock:
                    if thread in self.threads:
                        self.threads.remove(thread)
                    if thread in self.thread_map:
                        del self.thread_map[thread]
                    if tid in self.thread_info:
                        del self.thread_info[tid]
        if not self.thread_info:
            self.running = False
            self.log_event("All threads terminated, simulation stopped")

        self.root.after(50, self.update_ui)

if __name__ == "__main__":
    try:
        root = tk.Tk()
        root.geometry("900x700")  # Set initial window size
        app = ThreadSimulator(root)
        root.mainloop()  # Start the Tkinter event loop
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        messagebox.showerror("Fatal Error", f"Application failed to start: {e}")