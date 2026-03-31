"""
Main GUI application for the Autism Eye Tracking System.
Built with CustomTkinter for a modern dark-themed interface.

Uses a queue for thread→GUI communication (reliable across all platforms).
"""

import os
import sys
import threading
import time
import csv
import queue
from datetime import datetime

import cv2
import numpy as np
from PIL import Image, ImageTk

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

# CustomTkinter import
import customtkinter as ctk

from tracking.tracking import GazeTracker, GazeData


# ═══════════════════════════════════════════════════════════════════
# App colour scheme
# ═══════════════════════════════════════════════════════════════════
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")

BG_DARK   = "#1a1a2e"
BG_CARD   = "#16213e"
ACCENT    = "#0f3460"
GREEN     = "#00d474"
RED       = "#e94560"
YELLOW    = "#f0c040"
WHITE     = "#e0e0e0"
GRAY      = "#888888"


class AutismEyeApp(ctk.CTk):
    """Main application window."""

    def __init__(self):
        super().__init__()

        self.title("Autism Detection – Eye Tracking System")
        self.geometry("1100x750")
        self.minsize(900, 650)
        self.configure(fg_color=BG_DARK)

        # State
        self._tracking = False
        self._recording = False
        self._cap = None
        self._tracker = None
        self._csv_file = None
        self._csv_writer = None
        self._gaze = None
        self._frame_count = 0
        self._fps = 0
        self._fps_time = time.time()
        self._test_running = False
        self._last_test_results = None  # Store last test results

        # Thread-safe message queue for GUI updates
        self._msg_queue = queue.Queue()

        self._build_ui()
        self._update_loop_id = None

        # Start polling the message queue
        self._poll_queue()

    # ------------------------------------------------------------------
    # Message queue polling (runs on main thread, checks every 100ms)
    # ------------------------------------------------------------------
    def _poll_queue(self):
        """Process all pending messages from background threads."""
        try:
            while True:
                msg = self._msg_queue.get_nowait()
                action = msg.get("action")

                if action == "show_result":
                    self.lbl_result.configure(
                        text=msg["text"],
                        text_color=msg["color"]
                    )
                    # Bring window to front
                    self.deiconify()
                    self.lift()
                    self.focus_force()
                    self.update()

                elif action == "enable_button":
                    btn = getattr(self, msg["button"], None)
                    if btn:
                        btn.configure(state="normal")
                        if "text" in msg:
                            btn.configure(text=msg["text"])

                elif action == "show_window":
                    self.deiconify()
                    self.lift()
                    self.focus_force()

                elif action == "start_tracking":
                    self._on_start_tracking()

                elif action == "test_done":
                    self._test_running = False
                    self.btn_test.configure(state="normal", text="🧪  Run Screening Test")

        except queue.Empty:
            pass
        except Exception:
            pass

        # Schedule next poll
        self.after(100, self._poll_queue)

    def _send_msg(self, **kwargs):
        """Send a message to the GUI from any thread."""
        self._msg_queue.put(kwargs)

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        # Main grid: left (camera + gaze) | right (controls + info)
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ── Left panel ──
        left = ctk.CTkFrame(self, fg_color=BG_DARK)
        left.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=10)
        left.grid_rowconfigure(0, weight=4)
        left.grid_rowconfigure(1, weight=1)
        left.grid_columnconfigure(0, weight=1)

        # Camera preview
        cam_frame = ctk.CTkFrame(left, fg_color=BG_CARD, corner_radius=12)
        cam_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 5))
        cam_frame.grid_rowconfigure(0, weight=0)
        cam_frame.grid_rowconfigure(1, weight=1)
        cam_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(cam_frame, text="📹 Camera Preview", font=("Segoe UI", 16, "bold"),
                     text_color=WHITE).grid(row=0, column=0, pady=(10, 5))

        self.cam_label = ctk.CTkLabel(cam_frame, text="Camera not started",
                                       text_color=GRAY, font=("Segoe UI", 14))
        self.cam_label.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))

        # Gaze indicator (mini screen map)
        gaze_frame = ctk.CTkFrame(left, fg_color=BG_CARD, corner_radius=12)
        gaze_frame.grid(row=1, column=0, sticky="nsew", pady=(5, 0))
        gaze_frame.grid_rowconfigure(0, weight=0)
        gaze_frame.grid_rowconfigure(1, weight=1)
        gaze_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(gaze_frame, text="👁 Real-Time Gaze Indicator",
                     font=("Segoe UI", 14, "bold"),
                     text_color=WHITE).grid(row=0, column=0, pady=(8, 2))

        self.gaze_canvas_label = ctk.CTkLabel(gaze_frame, text="")
        self.gaze_canvas_label.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 8))

        # ── Right panel (scrollable) ──
        right = ctk.CTkScrollableFrame(self, fg_color=BG_DARK)
        right.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=10)
        right.grid_columnconfigure(0, weight=1)

        # Title
        ctk.CTkLabel(right, text="🧠 Control Panel", font=("Segoe UI", 16, "bold"),
                     text_color=WHITE).pack(pady=(5, 10))

        # --- Camera selector ---
        cam_sel_frame = ctk.CTkFrame(right, fg_color="transparent")
        cam_sel_frame.pack(fill="x", padx=10, pady=(0, 5))

        ctk.CTkLabel(cam_sel_frame, text="📷 Camera:", font=("Segoe UI", 12),
                     text_color=GRAY).pack(side="left", padx=(0, 5))

        self.camera_var = ctk.StringVar(value="0 – Laptop")
        self.camera_dropdown = ctk.CTkOptionMenu(
            cam_sel_frame,
            values=["0 – Laptop", "1 – Phone/External", "2 – Camera 2", "3 – Camera 3"],
            variable=self.camera_var,
            font=("Segoe UI", 11),
            height=28,
            fg_color=ACCENT,
            button_color=ACCENT,
            command=self._on_camera_changed
        )
        self.camera_dropdown.pack(side="left", fill="x", expand=True)

        # --- Tracking Buttons ---
        self.btn_start = ctk.CTkButton(
            right, text="▶  Start Tracking", font=("Segoe UI", 13, "bold"),
            fg_color=GREEN, text_color="#111", hover_color="#00b85c",
            height=36, corner_radius=10, command=self._on_start_tracking
        )
        self.btn_start.pack(fill="x", padx=10, pady=3)

        self.btn_stop = ctk.CTkButton(
            right, text="⏹  Stop Tracking", font=("Segoe UI", 13, "bold"),
            fg_color=RED, text_color="#fff", hover_color="#c03050",
            height=36, corner_radius=10, command=self._on_stop_tracking,
            state="disabled"
        )
        self.btn_stop.pack(fill="x", padx=10, pady=3)

        self.btn_record = ctk.CTkButton(
            right, text="⏺  Start Recording", font=("Segoe UI", 12),
            fg_color=ACCENT, hover_color="#1a4a80",
            height=32, corner_radius=10, command=self._on_toggle_recording,
            state="disabled"
        )
        self.btn_record.pack(fill="x", padx=10, pady=3)

        # Separator
        ctk.CTkFrame(right, height=2, fg_color=ACCENT).pack(fill="x", padx=10, pady=8)

        # --- Test buttons ---
        ctk.CTkLabel(right, text="🔬 Autism Screening", font=("Segoe UI", 14, "bold"),
                     text_color=WHITE).pack(pady=(0, 5))

        self.btn_calibrate = ctk.CTkButton(
            right, text="🎯  Calibrate", font=("Segoe UI", 12),
            fg_color=ACCENT, hover_color="#1a4a80",
            height=32, corner_radius=10, command=self._on_calibrate
        )
        self.btn_calibrate.pack(fill="x", padx=10, pady=3)

        self.btn_test = ctk.CTkButton(
            right, text="🧪  Run Screening Test", font=("Segoe UI", 13, "bold"),
            fg_color="#6c3fd4", hover_color="#5530a8",
            height=38, corner_radius=10, command=self._on_run_test
        )
        self.btn_test.pack(fill="x", padx=10, pady=3)

        self.btn_analyze = ctk.CTkButton(
            right, text="📊  Analyze Data", font=("Segoe UI", 12),
            fg_color=ACCENT, hover_color="#1a4a80",
            height=32, corner_radius=10, command=self._on_analyze
        )
        self.btn_analyze.pack(fill="x", padx=10, pady=3)

        # Separator
        ctk.CTkFrame(right, height=2, fg_color=ACCENT).pack(fill="x", padx=10, pady=8)

        # --- RESULTS (placed here so it's ALWAYS visible) ---
        self.result_frame = ctk.CTkFrame(right, fg_color=BG_CARD, corner_radius=10)
        self.result_frame.pack(fill="x", padx=10, pady=4)

        ctk.CTkLabel(self.result_frame, text="📋 Results", font=("Segoe UI", 13, "bold"),
                     text_color=WHITE).pack(anchor="w", padx=12, pady=(8, 2))

        self.lbl_result = ctk.CTkLabel(
            self.result_frame, text="No test run yet.\nClick 'Run Screening Test' to start.",
            font=("Segoe UI", 12), text_color=GRAY,
            wraplength=230, justify="left"
        )
        self.lbl_result.pack(anchor="w", padx=12, pady=(2, 10))

        # Separator
        ctk.CTkFrame(right, height=2, fg_color=ACCENT).pack(fill="x", padx=10, pady=8)

        # --- Status indicators (compact) ---
        status_frame = ctk.CTkFrame(right, fg_color=BG_CARD, corner_radius=10)
        status_frame.pack(fill="x", padx=10, pady=4)

        ctk.CTkLabel(status_frame, text="Status", font=("Segoe UI", 12, "bold"),
                     text_color=GRAY).pack(anchor="w", padx=12, pady=(6, 2))

        self.lbl_tracking = self._status_row(status_frame, "Tracking:", "Stopped", RED)
        self.lbl_recording = self._status_row(status_frame, "Recording:", "Off", GRAY)
        self.lbl_fps = self._status_row(status_frame, "FPS:", "—", GRAY)
        self.lbl_blinks = self._status_row(status_frame, "Blinks:", "—", GRAY)
        self.lbl_ear = self._status_row(status_frame, "EAR:", "—", GRAY)

        ctk.CTkFrame(status_frame, height=6, fg_color="transparent").pack()

        # Handle window close
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _status_row(self, parent, label, value, color):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=1)
        ctk.CTkLabel(row, text=label, font=("Segoe UI", 12),
                     text_color=GRAY, width=80, anchor="w").pack(side="left")
        val_lbl = ctk.CTkLabel(row, text=value, font=("Segoe UI", 12, "bold"),
                                text_color=color, anchor="w")
        val_lbl.pack(side="left")
        return val_lbl

    # ------------------------------------------------------------------
    # Camera & Tracking
    # ------------------------------------------------------------------
    def _get_camera_index(self):
        """Get the selected camera index from the dropdown."""
        val = self.camera_var.get()
        return int(val.split(" ")[0])

    def _on_camera_changed(self, value):
        """Called when camera dropdown changes. Update config globally."""
        idx = self._get_camera_index()
        config.CAMERA_INDEX = idx
        # If tracking, restart with new camera
        if self._tracking:
            self._on_stop_tracking()
            self._on_start_tracking()

    def _on_start_tracking(self):
        if self._tracking:
            return
        try:
            cam_idx = self._get_camera_index()
            config.CAMERA_INDEX = cam_idx
            self._cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
            if not self._cap.isOpened():
                self.lbl_result.configure(text="❌ Cannot open camera", text_color=RED)
                return
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_PREVIEW_W)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_PREVIEW_H)
            self._tracker = GazeTracker()
            self._tracking = True
            self._frame_count = 0
            self._fps_time = time.time()

            self.btn_start.configure(state="disabled")
            self.btn_stop.configure(state="normal")
            self.btn_record.configure(state="normal")
            self.lbl_tracking.configure(text="Active", text_color=GREEN)

            self._update_camera()
        except Exception as e:
            self.lbl_result.configure(text=f"❌ {e}", text_color=RED)

    def _on_stop_tracking(self):
        self._tracking = False

        if self._recording:
            self._on_toggle_recording()

        # Cancel the update loop FIRST
        if self._update_loop_id:
            self.after_cancel(self._update_loop_id)
            self._update_loop_id = None

        if self._cap:
            self._cap.release()
            self._cap = None
        if self._tracker:
            self._tracker.close()
            self._tracker = None

        self.btn_start.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        self.btn_record.configure(state="disabled")
        self.lbl_tracking.configure(text="Stopped", text_color=RED)
        self.lbl_fps.configure(text="—", text_color=GRAY)
        self.cam_label.configure(text="Camera stopped")

    def _update_camera(self):
        if not self._tracking or not self._cap:
            return

        ret, frame = self._cap.read()
        if not ret:
            self._update_loop_id = self.after(30, self._update_camera)
            return

        frame = cv2.flip(frame, 1)
        self._gaze = self._tracker.process_frame(frame)
        self._tracker.draw_debug(frame, self._gaze)

        # Record if active
        if self._recording and self._csv_writer and self._gaze:
            self._csv_writer.writerow(self._gaze.csv_row())

        # FPS calculation
        self._frame_count += 1
        elapsed = time.time() - self._fps_time
        if elapsed >= 1.0:
            self._fps = self._frame_count / elapsed
            self._frame_count = 0
            self._fps_time = time.time()
            self.lbl_fps.configure(text=f"{self._fps:.0f}", text_color=GREEN if self._fps > 15 else YELLOW)

        # Update blinks / EAR
        if self._tracker:
            self.lbl_blinks.configure(text=str(self._tracker.total_blinks), text_color=WHITE)
        if self._gaze:
            ear = (self._gaze.left_ear + self._gaze.right_ear) / 2 if self._gaze.left_ear else 0
            self.lbl_ear.configure(text=f"{ear:.2f}", text_color=WHITE)

        # Convert frame to Tkinter-compatible image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)

        # Scale to fit the label
        lw = self.cam_label.winfo_width() or config.CAMERA_PREVIEW_W
        lh = self.cam_label.winfo_height() or config.CAMERA_PREVIEW_H
        if lw > 50 and lh > 50:
            scale = min(lw / img.width, lh / img.height)
            new_w, new_h = int(img.width * scale), int(img.height * scale)
            if new_w > 0 and new_h > 0:
                img = img.resize((new_w, new_h), Image.LANCZOS)

        self._cam_photo = ImageTk.PhotoImage(img)
        self.cam_label.configure(image=self._cam_photo, text="")

        # Update gaze indicator
        self._update_gaze_indicator()

        self._update_loop_id = self.after(16, self._update_camera)

    def _update_gaze_indicator(self):
        """Draw a mini screen representation with gaze dot."""
        w, h = 280, 160
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        canvas[:] = (30, 35, 50)

        cv2.rectangle(canvas, (5, 5), (w - 5, h - 5), (80, 80, 100), 1)

        if self._gaze and self._gaze.iris_center and not self._gaze.blink:
            ix, iy = self._gaze.iris_center
            cam_w = config.CAMERA_PREVIEW_W
            cam_h = config.CAMERA_PREVIEW_H
            gx = int(5 + (ix / cam_w) * (w - 10))
            gy = int(5 + (iy / cam_h) * (h - 10))
            gx = max(8, min(gx, w - 8))
            gy = max(8, min(gy, h - 8))

            cv2.circle(canvas, (gx, gy), 12, (0, 60, 30), -1)
            cv2.circle(canvas, (gx, gy), 7, (0, 200, 100), -1)
            cv2.circle(canvas, (gx, gy), 3, (200, 255, 200), -1)
        elif self._gaze and self._gaze.blink:
            cv2.putText(canvas, "BLINK", (w // 2 - 35, h // 2 + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 1)

        img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        self._gaze_photo = ImageTk.PhotoImage(img)
        self.gaze_canvas_label.configure(image=self._gaze_photo)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------
    def _on_toggle_recording(self):
        if not self._recording:
            os.makedirs(config.RAW_DATA_DIR, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(config.RAW_DATA_DIR, f"gaze_{ts}.csv")
            self._csv_file = open(path, "w", newline="")
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow(GazeData.csv_header())
            self._recording = True
            self.btn_record.configure(text="⏹  Stop Recording", fg_color=RED)
            self.lbl_recording.configure(text="Recording", text_color=GREEN)
        else:
            self._recording = False
            if self._csv_file:
                self._csv_file.close()
                self._csv_file = None
                self._csv_writer = None
            self.btn_record.configure(text="⏺  Start Recording", fg_color=ACCENT)
            self.lbl_recording.configure(text="Off", text_color=GRAY)

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------
    def _on_calibrate(self):
        was_tracking = self._tracking
        if was_tracking:
            self._on_stop_tracking()

        self.btn_calibrate.configure(state="disabled")
        self.lbl_result.configure(
            text="Running calibration...\nFollow the dots on screen.",
            text_color=YELLOW
        )

        # HIDE the GUI window so it doesn't overlap with calibration
        self.withdraw()

        def _run():
            try:
                time.sleep(0.5)
                from tracking.calibration import run_calibration
                run_calibration()

                self._send_msg(
                    action="show_result",
                    text="✅ Calibration complete!\n\nYou can now run the\nscreening test.",
                    color=GREEN
                )
            except Exception as e:
                self._send_msg(action="show_result", text=f"❌ Calibration failed: {e}", color=RED)
            finally:
                self._send_msg(action="enable_button", button="btn_calibrate")
                self._send_msg(action="show_window")
                if was_tracking:
                    time.sleep(0.5)
                    self._send_msg(action="start_tracking")

        threading.Thread(target=_run, daemon=True).start()

    # ------------------------------------------------------------------
    # Run Autism Screening Test
    # ------------------------------------------------------------------
    def _on_run_test(self):
        if self._test_running:
            return

        # Ask for subject name BEFORE starting
        dialog = ctk.CTkInputDialog(
            text="Enter the subject's name:\n(This will appear in the report)",
            title="Subject Name"
        )
        subject_name = dialog.get_input()
        if not subject_name or subject_name.strip() == "":
            self.lbl_result.configure(text="Test cancelled.\nNo name entered.", text_color=GRAY)
            return
        subject_name = subject_name.strip()

        was_tracking = self._tracking
        if was_tracking:
            self._on_stop_tracking()

        self._test_running = True
        self.btn_test.configure(state="disabled", text="⏳ Test Running...")
        self.lbl_result.configure(
            text=f"Preparing test for:\n{subject_name}\n\nStimulus images will appear.\nLook naturally at each image.",
            text_color=YELLOW
        )

        # HIDE the GUI window so it doesn't overlap with test
        self.withdraw()

        def _run():
            try:
                time.sleep(0.5)

                from app.test_session import run_test_session
                from app.classifier import AutismGazeClassifier
                from app.report import generate_report

                results = run_test_session()
                if results is None:
                    self._send_msg(action="show_result", text="Test was cancelled.", color=RED)
                    self._send_msg(action="test_done")
                    self._send_msg(action="show_window")
                    return

                # Classify
                classifier = AutismGazeClassifier()
                classification = classifier.classify(results)

                # Generate report with subject name
                generate_report(results, classification, subject_name=subject_name)

                # Store results
                self._last_test_results = classification

                # Build result text
                risk = classification.get("risk_score", 0)
                cls_label = classification.get("classification", "N/A")
                face_ratio = classification.get("face_attention_ratio", 0)
                blinks = results.get("total_blinks", 0)

                if cls_label == "Typical":
                    result_color = GREEN
                elif cls_label == "Moderate Risk":
                    result_color = YELLOW
                else:
                    result_color = RED

                result_text = (
                    f"Subject: {subject_name}\n"
                    f"Classification: {cls_label}\n"
                    f"Risk Score: {risk:.1f}%\n"
                    f"Face Attention: {face_ratio:.1%}\n"
                    f"Blinks: {blinks}\n\n"
                    f"Report saved to\nexperiments/reports/\n\n"
                    f"⚠️ Research tool only.\n"
                    f"Not a medical diagnosis."
                )

                self._send_msg(action="show_result", text=result_text, color=result_color)

            except Exception as e:
                self._send_msg(action="show_result", text=f"❌ Test failed: {e}", color=RED)
            finally:
                self._send_msg(action="test_done")
                self._send_msg(action="show_window")
                if was_tracking:
                    time.sleep(0.5)
                    self._send_msg(action="start_tracking")

        threading.Thread(target=_run, daemon=True).start()

    # ------------------------------------------------------------------
    # Analyze
    # ------------------------------------------------------------------
    def _on_analyze(self):
        self.btn_analyze.configure(state="disabled")
        self.lbl_result.configure(text="Analyzing data...", text_color=YELLOW)

        def _run():
            try:
                from data.processing import process_raw
                from data.features import extract_and_save
                from app.heatmap import generate_heatmap

                df, proc_path = process_raw()
                feats, _ = extract_and_save(proc_path)
                hm_path = generate_heatmap(proc_path)

                summary = (
                    f"✅ Analysis complete!\n\n"
                    f"Fixations: {feats.get('fixation_count', 0)}\n"
                    f"Saccades: {feats.get('saccade_count', 0)}\n"
                    f"Scanpath: {feats.get('scanpath_length_px', 0):.0f} px\n"
                    f"Heatmap saved."
                )
                self._send_msg(action="show_result", text=summary, color=GREEN)
            except Exception as e:
                self._send_msg(action="show_result", text=f"❌ Analysis failed: {e}", color=RED)
            finally:
                self._send_msg(action="enable_button", button="btn_analyze")

        threading.Thread(target=_run, daemon=True).start()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _on_close(self):
        self._tracking = False
        if self._recording:
            self._on_toggle_recording()
        if self._update_loop_id:
            self.after_cancel(self._update_loop_id)
        if self._cap:
            self._cap.release()
        if self._tracker:
            self._tracker.close()
        self.destroy()


def launch_gui():
    """Entry point to launch the GUI."""
    app = AutismEyeApp()
    app.mainloop()


if __name__ == "__main__":
    launch_gui()
