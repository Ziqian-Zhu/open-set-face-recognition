"""Polished Tkinter desktop interface for camera recognition and enrollment."""

from __future__ import annotations

import queue
import threading
import time
import sqlite3
import math
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

from .camera import CameraDevice, CameraManager
from .config import AppConfig
from .models import FaceObservation, PreparedSample, RecognitionResult
from .recognizer import MultiFrameVoter
from .service import FaceComparisonSystem
from .enrollment import EnrollmentSession, TrackContinuity
from .worker import InferenceWorker
from .motion import MotionScheduler, mix_color


class FaceCompareApp(tk.Tk):
    """Quiet, image-first desktop workspace with progressive controls."""

    BACKGROUND = "#F2F4F5"
    CARD = "#FFFFFF"
    CARD_ALT = "#EDF1F2"
    TEXT = "#192B32"
    MUTED = "#667880"
    ACCENT = "#087F73"
    ACCENT_DARK = "#076C63"
    BORDER = "#DFE6E8"
    PREVIEW = "#11232C"
    WARNING = "#9D660D"
    DANGER = "#BB414C"
    SUCCESS = "#087F73"

    def __init__(self, config: AppConfig, project_root: str | Path) -> None:
        super().__init__()
        self.withdraw()  # Build the complete workspace before showing the first frame.
        self.config = config
        self.system = FaceComparisonSystem(config, project_root)
        self.camera = CameraManager(config.camera)
        self._video_source = None
        self._source_requested = False
        self.voter = MultiFrameVoter(config.recognition)
        self.continuity = TrackContinuity(self.system.extractor)
        self.session = EnrollmentSession()
        self.worker = InferenceWorker(self.system)
        self.generation = 0
        self._last_analysis = 0.0
        self._current_frame_time = 0.0
        self._sample_photos = []
        self.current_frame = None
        self.current_observations: list[FaceObservation] = []
        self.enrollment_samples = self.session.samples
        self.frame_number = 0
        self.last_stable_key: tuple[bool, str | None] | None = None
        self._preview_photo: ImageTk.PhotoImage | None = None
        self._devices: dict[str, CameraDevice] = {}
        self._device_queue: queue.Queue[tuple[list[CameraDevice] | None, str | None]] = queue.Queue()
        self._scanning = False
        self._motion_state = None
        self._feedback_until = 0.
        self.motion = MotionScheduler(self, reduced=config.ui.reduce_motion)

        self.title(config.ui.title)
        width, height = min(1360, self.winfo_screenwidth()-64), min(860, self.winfo_screenheight()-90)
        self.geometry(f"{width}x{height}+{max(0, (self.winfo_screenwidth()-width)//2)}+{max(28, (self.winfo_screenheight()-height)//2)}")
        self.minsize(960, 640)
        self.configure(bg=self.BACKGROUND)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._configure_styles()
        self._build_variables()
        self._build_layout()
        self._set_motion_state("idle")
        self._refresh_database_panel()
        self.update_idletasks()
        self.deiconify()
        # Let the native window map and draw before scheduling background setup.
        self.after(150, self._begin_camera_scan)
        self.after(180, self._tick)

    def _configure_styles(self) -> None:
        """Define the small design system used throughout the window."""

        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        default_font = ("PingFang SC", 11)
        style.configure("TFrame", background=self.BACKGROUND)
        style.configure("Card.TFrame", background=self.CARD)
        style.configure("AltCard.TFrame", background=self.CARD_ALT)
        style.configure("TLabel", background=self.BACKGROUND, foreground=self.TEXT, font=default_font)
        style.configure("Card.TLabel", background=self.CARD, foreground=self.TEXT, font=default_font)
        style.configure("Muted.Card.TLabel", background=self.CARD, foreground=self.MUTED, font=("PingFang SC", 10))
        style.configure("Title.TLabel", background=self.BACKGROUND, foreground=self.TEXT, font=("PingFang SC", 24, "bold"))
        style.configure("Subtitle.TLabel", background=self.BACKGROUND, foreground=self.MUTED, font=("PingFang SC", 10))
        style.configure("Motion.TCheckbutton", background=self.BACKGROUND, foreground=self.MUTED,
                        font=("PingFang SC", 9))
        style.configure("Section.Card.TLabel", background=self.CARD, foreground=self.TEXT, font=("PingFang SC", 13, "bold"))
        style.configure("Result.Card.TLabel", background=self.CARD, foreground=self.TEXT, font=("PingFang SC", 20, "bold"))
        style.configure("Metric.Card.TLabel", background=self.CARD, foreground=self.ACCENT, font=("Menlo", 12, "bold"))
        style.configure(
            "TButton",
            background=self.CARD_ALT,
            foreground=self.TEXT,
            borderwidth=0,
            relief="flat",
            focusthickness=2,
            focuscolor=self.ACCENT,
            padding=(14, 10),
            font=default_font,
        )
        style.map("TButton", background=[("disabled", "#EEF1F2"), ("pressed", "#D7E1E3"), ("active", "#E1E8EA")],
                  foreground=[("disabled", "#9AA7AD")])
        style.configure(
            "Accent.TButton",
            background=self.ACCENT_DARK,
            foreground="#FFFFFF",
            font=("PingFang SC", 11, "bold"),
        )
        style.map("Accent.TButton", background=[("disabled", "#DFE8E6"), ("pressed", "#055C54"), ("active", self.ACCENT)],
                  foreground=[("disabled", "#81958F"), ("!disabled", "#FFFFFF")])
        style.configure("Ghost.TButton", background=self.CARD, foreground=self.MUTED, padding=(10, 7))
        style.configure("Danger.TButton", background="#FBECEE", foreground=self.DANGER)
        style.map("Danger.TButton", background=[("disabled", self.CARD_ALT), ("active", "#F6DCE0")],
                  foreground=[("disabled", "#9AA7AD"), ("!disabled", self.DANGER)])
        style.configure("TCombobox", fieldbackground=self.CARD_ALT, background=self.CARD_ALT,
                        foreground=self.TEXT, arrowcolor=self.MUTED, bordercolor=self.BORDER, padding=8)
        style.map("TCombobox", fieldbackground=[("readonly", self.CARD_ALT), ("disabled", self.CARD_ALT)],
                  foreground=[("readonly", self.TEXT), ("disabled", self.MUTED)])
        style.configure("TEntry", fieldbackground="#F7F9FA", foreground=self.TEXT, insertcolor=self.TEXT,
                        bordercolor=self.BORDER, lightcolor=self.BORDER, darkcolor=self.BORDER, padding=10)
        style.map("TEntry", bordercolor=[("focus", self.ACCENT)], fieldbackground=[("disabled", self.CARD_ALT)])
        style.configure("Horizontal.TProgressbar", troughcolor=self.CARD_ALT, background=self.ACCENT,
                        borderwidth=0, thickness=5, lightcolor=self.ACCENT, darkcolor=self.ACCENT)
        style.configure("TSeparator", background=self.BORDER)
        style.configure("TNotebook", background=self.CARD, borderwidth=0, tabmargins=(12, 10, 12, 0))
        style.configure("TNotebook.Tab", background=self.CARD, foreground=self.MUTED,
                        padding=(14, 10), font=("PingFang SC", 11), borderwidth=0)
        style.map("TNotebook.Tab", background=[("selected", "#E7F3EF"), ("active", self.CARD_ALT)],
                  foreground=[("selected", self.ACCENT)])
        style.configure("Treeview", background=self.CARD, fieldbackground=self.CARD, foreground=self.TEXT,
                        rowheight=36, borderwidth=0, font=default_font)
        style.configure("Treeview.Heading", background=self.CARD_ALT, foreground=self.MUTED,
                        relief="flat", padding=10, font=("PingFang SC", 10))
        style.map("Treeview", background=[("selected", "#E7F3EF")], foreground=[("selected", self.ACCENT)])
        self.option_add("*TCombobox*Listbox.background", self.CARD)
        self.option_add("*TCombobox*Listbox.foreground", self.TEXT)
        self.option_add("*TCombobox*Listbox.selectBackground", "#E7F3EF")
        self.option_add("*TCombobox*Listbox.selectForeground", self.ACCENT)

    def _build_variables(self) -> None:
        """Create observable strings shared by widgets."""

        self.camera_choice = tk.StringVar(value="正在扫描摄像头…")
        self.person_name = tk.StringVar()
        self.connection_text = tk.StringVar(value="未连接")
        self.result_name = tk.StringVar(value="准备好，开始识别")
        self.result_detail = tk.StringVar(value="连接摄像头后，系统将自动检测并确认身份。")
        self.quality_text = tk.StringVar(value="尚无画面 · 光照、清晰度与人脸大小将在这里显示")
        self.sample_text = tk.StringVar(value=f"0 / {self.config.ui.enrollment_target_samples}")
        self.database_text = tk.StringVar()
        self.threshold_text = tk.StringVar()
        self.footer_text = tk.StringVar(value="系统就绪 · 所有识别均在本机完成")
        self.reduce_motion = tk.BooleanVar(value=self.config.ui.reduce_motion)

    def _build_layout(self) -> None:
        """Assemble header, preview, controls, and status footer."""

        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        header = ttk.Frame(self, padding=(28, 22, 28, 20))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="F A C E  /  S T U D I O", foreground=self.ACCENT,
                  font=("Menlo", 10, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 6))
        ttk.Label(header, text="人脸识别工作台", style="Title.TLabel").grid(row=1, column=0, sticky="w")
        ttk.Label(
            header,
            text="从清晰的一张脸，到可靠的一次确认。",
            style="Subtitle.TLabel",
        ).grid(row=2, column=0, sticky="w", pady=(5, 0))
        badge = tk.Label(
            header,
            text="●  本地处理  /  无需上传",
            bg="#E2F0E9",
            fg=self.ACCENT,
            font=("PingFang SC", 10),
            padx=14,
            pady=9,
        )
        badge.grid(row=0, column=1, sticky="e")
        self.manage_saved_button = ttk.Button(header, text="管理 / 删除已保存人员", command=self._show_saved_people)
        self.manage_saved_button.grid(row=1, column=1, sticky="e", pady=(4, 4))
        ttk.Label(header, textvariable=self.database_text, style="Subtitle.TLabel").grid(row=2, column=1, sticky="e")

        content = ttk.Frame(self, padding=(28, 4, 28, 14))
        content.grid(row=1, column=0, sticky="nsew")
        content.columnconfigure(0, weight=1)
        content.columnconfigure(1, minsize=360)
        content.rowconfigure(0, weight=1)

        preview_card = ttk.Frame(content, style="Card.TFrame", padding=16)
        preview_card.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        preview_card.columnconfigure(0, weight=1)
        preview_card.rowconfigure(1, weight=1)
        preview_header = ttk.Frame(preview_card, style="Card.TFrame")
        preview_header.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        preview_header.columnconfigure(0, weight=1)
        ttk.Label(preview_header, text="实时视窗", style="Section.Card.TLabel").grid(row=0, column=0, sticky="w")
        self.status_beacon = tk.Canvas(preview_header, width=24, height=24, bg=self.CARD, highlightthickness=0)
        self.status_beacon.grid(row=0, column=1, sticky="e", padx=(8, 4))
        ttk.Label(preview_header, textvariable=self.connection_text, style="Muted.Card.TLabel").grid(row=0, column=2, sticky="e")
        self.preview = tk.Label(
            preview_card,
            text="",
            bg=self.PREVIEW,
            fg=self.MUTED,
            font=("PingFang SC", 14),
            width=1,
            height=1,
        )
        self.preview.grid(row=1, column=0, sticky="nsew")
        self.empty_preview = tk.Canvas(self.preview, bg=self.PREVIEW, highlightthickness=0)
        self.empty_preview.place(relwidth=1, relheight=1)
        self.empty_preview.bind("<Configure>", self._draw_empty_preview)
        ttk.Label(preview_card, text=f"{self.system.engine_label}   /   多人独立确认 · 未知拒识",
                  style="Muted.Card.TLabel").grid(row=2, column=0, sticky="w", pady=(12, 2))
        ttk.Separator(preview_card).grid(row=3, column=0, sticky="ew", pady=(12, 0))
        self._build_result_card(preview_card).grid(row=4, column=0, sticky="ew")

        side_container = ttk.Frame(content)
        side_container.grid(row=0, column=1, sticky="nsew")
        side_container.rowconfigure(0, weight=1)
        side_container.columnconfigure(0, weight=1)
        canvas = tk.Canvas(side_container, width=354, bg=self.BACKGROUND, highlightthickness=0)
        self._controls_canvas = canvas
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(side_container, orient="vertical", command=canvas.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        def update_scrollbar(first, last):
            scrollbar.set(first, last)
            if float(first) <= 0 and float(last) >= 1:
                scrollbar.grid_remove()
            else:
                scrollbar.grid()
        canvas.configure(yscrollcommand=update_scrollbar)
        side = ttk.Frame(canvas)
        window = canvas.create_window((0, 0), window=side, anchor="nw")
        side.bind("<Configure>", lambda event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(window, width=event.width))
        side.columnconfigure(0, weight=1)
        self._build_camera_card(side).grid(row=0, column=0, sticky="ew", pady=(0, 14))
        self.workspace_tabs = ttk.Notebook(side)
        self.workspace_tabs.grid(row=1, column=0, sticky="ew")
        enrollment = self._build_enrollment_card(self.workspace_tabs)
        library = self._build_database_card(self.workspace_tabs)
        self.workspace_tabs.add(enrollment, text="录入人员")
        self.workspace_tabs.add(library, text="已保存人员")
        tracks_page = ttk.Frame(self.workspace_tabs, padding=16, style="Card.TFrame")
        self.workspace_tabs.add(tracks_page, text="视频轨迹")
        ttk.Label(tracks_page, text="每张脸，独立确认", style="Section.Card.TLabel").pack(anchor="w")
        ttk.Label(tracks_page, text="轨迹编号不是身份编号；录入时仍须单人。", style="Muted.Card.TLabel", wraplength=290).pack(anchor="w", pady=(8, 12))
        list_frame = ttk.Frame(tracks_page, style="Card.TFrame")
        list_frame.pack(fill="x")
        self.track_list = ttk.Treeview(list_frame, columns=("track", "identity", "state"), show="headings", height=5)
        for key, title, width in (("track", "轨迹", 44), ("identity", "身份", 110), ("state", "状态", 95)):
            self.track_list.heading(key, text=title)
            self.track_list.column(key, width=width, minwidth=40)
        self.track_list.pack(side="left", fill="x", expand=True)
        track_scroll = ttk.Scrollbar(list_frame, orient="vertical", command=self.track_list.yview)
        track_scroll.pack(side="right", fill="y")
        self.track_list.configure(yscrollcommand=track_scroll.set)
        ttk.Label(tracks_page, text="多人交叉或身份冲突时重新确认。\n尚未加入活体检测，照片/屏幕可能通过。", style="Muted.Card.TLabel", wraplength=290).pack(anchor="w", pady=10)
        ttk.Label(side, text="01  连接相机    →    02  采集样本    →    03  保存",
                  style="Subtitle.TLabel", font=("PingFang SC", 9)).grid(row=2, column=0, sticky="w", pady=14)

        def scroll_controls(event):
            widget = self.winfo_containing(event.x_root, event.y_root)
            if widget is not None and str(widget).startswith(str(side_container)):
                canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")
        self.bind("<MouseWheel>", scroll_controls, add="+")

        footer = ttk.Frame(self, padding=(28, 6, 28, 14))
        footer.grid(row=2, column=0, sticky="ew")
        footer.columnconfigure(0, weight=1)
        footer_label = ttk.Label(footer, textvariable=self.footer_text, style="Subtitle.TLabel", wraplength=700)
        self.footer_label = footer_label
        footer_label.grid(row=0, column=0, sticky="w")
        footer.bind("<Configure>", lambda event: footer_label.configure(wraplength=max(350, event.width-370)))
        self.reduce_motion_button = ttk.Checkbutton(footer, text="减少动画", variable=self.reduce_motion,
                                                    command=self._toggle_motion, style="Motion.TCheckbutton")
        self.reduce_motion_button.grid(row=0, column=1, sticky="e", padx=12)
        ttk.Label(footer, text="隐私优先 · ESC 停止", style="Subtitle.TLabel").grid(row=0, column=2, sticky="e")
        self.bind("<Escape>", lambda _event: self._stop_camera())

    def _set_motion_state(self, state, *, force=False):
        if state == self._motion_state and not force:
            return
        self._motion_state = state
        color = {"idle": self.MUTED, "connecting": self.ACCENT, "searching": self.ACCENT,
                 "confirming": self.WARNING, "known": self.SUCCESS, "unknown": self.DANGER,
                 "quality": self.WARNING, "mixed": self.WARNING, "error": self.DANGER}[state]
        busy = state in ("connecting", "searching", "confirming")

        def paint(progress):
            canvas = self.status_beacon
            canvas.delete("all")
            canvas.create_oval(3, 3, 21, 21, outline=self.BORDER, width=2)
            if busy:
                canvas.create_arc(3, 3, 21, 21, start=90-360*progress, extent=95,
                                  style="arc", outline=color, width=2)
            else:
                canvas.create_oval(3, 3, 21, 21, outline=mix_color(self.BORDER, color, .35), width=2)
            radius = 3 if busy or state == "idle" else 3+1.5*math.sin(math.pi*progress)
            canvas.create_oval(12-radius, 12-radius, 12+radius, 12+radius, fill=color, outline="")

        # Finite confirmation pulse; only actual ongoing work gets a spinner.
        self.motion.start("status", 1.15 if busy else .4, paint, repeat=busy)

    def _toggle_motion(self):
        self.motion.set_reduced(self.reduce_motion.get())
        self._set_motion_state(self._motion_state or "idle", force=True)

    def _feedback(self, tone="success"):
        self._feedback_until = time.monotonic()+3.
        color = {"success": "#DDEFE8", "warning": "#FFF0D7", "error": "#F9E2E5"}[tone]
        self.motion.start("feedback", .9, lambda t: self.footer_label.configure(
            background=mix_color(color, self.BACKGROUND, t)))

    def _animate_sample_progress(self, target):
        initial = float(self.sample_progress.cget("value"))
        self.motion.start("samples", .24, lambda t: self.sample_progress.configure(value=initial+(target-initial)*t))

    def _build_camera_card(self, parent: ttk.Frame) -> ttk.Frame:
        card = ttk.Frame(parent, style="Card.TFrame", padding=16)
        card.columnconfigure(0, weight=1)
        ttk.Label(card, text="设备连接", style="Section.Card.TLabel").grid(row=0, column=0, columnspan=2, sticky="w")
        self.camera_combo = ttk.Combobox(card, state="readonly", textvariable=self.camera_choice, width=18)
        self.camera_combo.grid(row=1, column=0, sticky="ew", pady=(12, 10), padx=(0, 8))
        self.scan_button = ttk.Button(card, text="刷新", command=self._begin_camera_scan)
        self.scan_button.grid(row=1, column=1, sticky="ew", pady=(12, 10))
        self.start_button = ttk.Button(card, text="开启摄像头", style="Accent.TButton", command=self._toggle_camera)
        self.start_button.grid(row=2, column=0, sticky="ew", padx=(0, 8))
        self.open_video_button = ttk.Button(card, text="打开视频", command=self._open_video)
        self.open_video_button.grid(row=2, column=1, sticky="ew")
        return card

    def _build_result_card(self, parent: ttk.Frame) -> ttk.Frame:
        card = ttk.Frame(parent, style="Card.TFrame", padding=(0, 14, 0, 0))
        card.columnconfigure(0, weight=1)
        ttk.Label(card, text="识别结果  /  IDENTITY", style="Muted.Card.TLabel").grid(row=0, column=0, sticky="w")
        self.result_label = ttk.Label(card, textvariable=self.result_name, style="Result.Card.TLabel")
        self.result_label.grid(row=1, column=0, sticky="w", pady=(4, 4))
        detail = ttk.Label(card, textvariable=self.result_detail, style="Muted.Card.TLabel", wraplength=500)
        detail.grid(row=2, column=0, sticky="w")
        quality = ttk.Label(card, textvariable=self.quality_text, style="Muted.Card.TLabel", wraplength=500)
        quality.grid(row=3, column=0, sticky="w", pady=(8, 0))
        def wrap_result(event):
            for label in (self.result_label, detail, quality):
                label.configure(wraplength=max(240, event.width-8))
        card.bind("<Configure>", wrap_result)
        return card

    def _build_enrollment_card(self, parent: ttk.Frame) -> ttk.Frame:
        card = ttk.Frame(parent, style="Card.TFrame", padding=16)
        card.columnconfigure(0, weight=1)
        ttk.Label(card, text="建立人员档案", style="Section.Card.TLabel").grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(card, text="人员姓名 · 采集后自动锁定", style="Muted.Card.TLabel").grid(row=1, column=0, columnspan=2, sticky="w", pady=(14, 0))
        self.name_entry = ttk.Entry(card, textvariable=self.person_name)
        self.name_entry.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(4, 9))
        self.sample_progress = ttk.Progressbar(
            card,
            maximum=self.config.ui.enrollment_target_samples,
            value=0,
        )
        self.sample_progress.grid(row=3, column=0, sticky="ew", pady=(12, 8), padx=(0, 8))
        ttk.Label(card, textvariable=self.sample_text, style="Metric.Card.TLabel").grid(row=3, column=1, sticky="e")
        self.capture_button = ttk.Button(card, text="采集一张", command=self._capture_enrollment_sample)
        self.capture_button.grid(row=4, column=0, sticky="ew", padx=(0, 8))
        self.save_button = ttk.Button(card, text="保存入库", style="Accent.TButton", command=self._save_enrollment)
        self.save_button.grid(row=4, column=1, sticky="ew")
        self.guide_text = tk.StringVar(value="第1步：输入姓名，正对镜头，画质合格后采集")
        ttk.Label(card, textvariable=self.guide_text, style="Muted.Card.TLabel", wraplength=292).grid(row=5, column=0, columnspan=2, sticky="w", pady=(14, 10))
        self.thumbnail_frame = ttk.Frame(card, style="Card.TFrame")
        self.thumbnail_frame.grid(row=6, column=0, columnspan=2, sticky="w")
        ttk.Button(card, text="清空未保存样本（不删除已保存人员）", style="Ghost.TButton", command=self._clear_enrollment).grid(row=7, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        return card

    def _build_database_card(self, parent: ttk.Frame) -> ttk.Frame:
        card = ttk.Frame(parent, style="Card.TFrame", padding=16)
        card.columnconfigure(0, weight=1)
        ttk.Label(card, text="已保存的人员", style="Section.Card.TLabel").grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(card, textvariable=self.database_text, style="Card.TLabel").grid(row=1, column=0, sticky="w", pady=(11, 2))
        ttk.Button(card, text="展开列表", style="Ghost.TButton", command=self._manage_people).grid(row=1, column=1, sticky="e")
        self.delete_saved_button = ttk.Button(card, text="先在下方选择人员", style="Danger.TButton",
                                             state="disabled", command=self._delete_selected_saved)
        self.delete_saved_button.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 10))
        list_frame = ttk.Frame(card, style="Card.TFrame")
        list_frame.grid(row=3, column=0, columnspan=2, sticky="ew")
        list_frame.columnconfigure(0, weight=1)
        self.saved_people_list = ttk.Treeview(list_frame, columns=("name", "count"), show="headings", height=3, selectmode="browse")
        self.saved_people_list.heading("name", text="姓名")
        self.saved_people_list.heading("count", text="样本数")
        self.saved_people_list.column("name", width=190, minwidth=100)
        self.saved_people_list.column("count", width=60, minwidth=55, stretch=False, anchor="center")
        self.saved_people_list.grid(row=0, column=0, sticky="ew")
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.saved_people_list.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.saved_people_list.configure(yscrollcommand=scrollbar.set)
        self.saved_people_list.bind("<<TreeviewSelect>>", lambda _event: self._update_saved_selection())
        self.saved_hint = tk.StringVar()
        ttk.Label(card, textvariable=self.saved_hint, style="Muted.Card.TLabel", wraplength=292).grid(row=4, column=0, columnspan=2, sticky="w", pady=(10, 8))
        ttk.Label(card, text="删除后不再参与识别；原照片和特征保留备份。\n这里不会清空你尚未保存的采集样本。",
                  style="Muted.Card.TLabel", wraplength=292).grid(row=5, column=0, columnspan=2, sticky="w")
        ttk.Separator(card).grid(row=6, column=0, columnspan=2, sticky="ew", pady=12)
        ttk.Label(card, textvariable=self.threshold_text, style="Muted.Card.TLabel", wraplength=180).grid(row=7, column=0, sticky="w")
        ttk.Button(card, text="阈值说明" if self.config.engine.backend == "sface" else "库内校准", command=self._calibrate).grid(row=7, column=1, sticky="e")
        ttk.Label(card, text=f"当前人脸库：{self.config.storage.database_directory}", style="Muted.Card.TLabel").grid(row=8, column=0, columnspan=2, sticky="w", pady=(8, 0))
        return card

    def _show_saved_people(self):
        self._refresh_database_panel()
        self.workspace_tabs.select(1)
        self._controls_canvas.yview_moveto(0)
        self.saved_people_list.focus_set()

    def _update_saved_selection(self):
        selected = self.saved_people_list.selection()
        if selected:
            name = self.saved_people_list.item(selected[0], "values")[0]
            short_name = name if len(name) <= 12 else name[:12]+"…"
            self.delete_saved_button.configure(state="normal", text=f"删除「{short_name}」")
        else:
            self.delete_saved_button.configure(state="disabled", text="先在下方选择人员")

    def _delete_selected_saved(self):
        selected = self.saved_people_list.selection()
        if selected:
            self._delete_saved_people(selected[0])

    def _delete_saved_people(self, person_id=None, *, parent=None):
        """Remove saved templates only; never discard unsaved enrollment work."""
        people = self.system.database.list_people()
        targets = [person for person in people if person_id is None or person.person_id == person_id]
        if not targets:
            return False
        description = f"「{targets[0].name}」" if person_id is not None else f"全部 {len(targets)} 名人员"
        count = sum(person.sample_count for person in targets)
        if not messagebox.askyesno("确认删除已保存人员", f"删除{description}及其 {count} 张识别样本？\n\n"
                                  "删除后立即停止对这些记录的匹配，可重新录入。\n"
                                  "原照片和特征仍保留在本地备份中，不是永久擦除。\n"
                                  "本次尚未保存的样本不受影响。", parent=parent or self, default="no"):
            return False
        try:
            with self.system.lock:
                backup = self.system.database.archive_people(person_id)
            self.voter.reset()
            self.continuity.reset()
            self.generation += 1
            self._last_analysis = 0.0
            self.last_stable_key = None
            self.current_observations = []
            self._refresh_tracks([])
            self.result_name.set("人员库已更新")
            self.result_label.configure(foreground=self.TEXT)
            self.result_detail.set("已删除的记录不再参与识别，等待新画面确认")
            self._set_motion_state("searching" if self._source_requested else "idle")
            self._refresh_database_panel()
            self.footer_text.set(f"已删除{description} · 可重新录入 · 备份：{backup.name}")
            self._feedback("warning")
            return True
        except (OSError, RuntimeError, ValueError, sqlite3.Error) as error:
            messagebox.showerror("删除失败", str(error), parent=parent or self)
            return False

    def _draw_empty_preview(self, _event=None):
        """Resolution-independent idle artwork, not a simulated camera feed."""

        canvas = self.empty_preview
        canvas.delete("all")
        w, h = canvas.winfo_width(), canvas.winfo_height()
        if w < 10 or h < 10:
            return
        compact = h < 280
        cx, cy = w/2, h*(.33 if compact else .40)
        radius = min(74, h*(.15 if compact else .20))
        for r in (radius+26, radius+48):
            canvas.create_oval(cx-r, cy-r, cx+r, cy+r, outline="#203740", width=1)
        a = radius
        for sx, sy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
            x, y = cx+sx*a, cy+sy*a
            canvas.create_line(x-sx*19, y, x, y, x, y-sy*19,
                               fill="#67C7B3", width=2, capstyle="round", joinstyle="round")
        canvas.create_oval(cx-a*.28, cy-a*.58, cx+a*.28, cy-a*.02, outline="#91AAB2", width=2)
        canvas.create_arc(cx-a*.58, cy+a*.14, cx+a*.58, cy+a*1.05,
                          start=0, extent=180, style="arc", outline="#91AAB2", width=2)
        canvas.create_text(22, 22, text="CAMERA  /  STANDBY", anchor="nw", fill="#829DA7", font=("Menlo", 9))
        text_y = cy+a+(30 if compact else 54)
        canvas.create_text(cx, text_y, text="让镜头认识你", fill="#E7F0F2", font=("PingFang SC", 16 if compact else 19, "bold"))
        canvas.create_text(cx, text_y+28, text="在右侧开启摄像头，实时画面将在这里呈现", fill="#91AAB2", font=("PingFang SC", 10), width=max(220, w-60))
        if h > 380:
            canvas.create_text(cx, h-26, text="本机识别   /   五点对齐   /   多帧确认", fill="#66838E", font=("PingFang SC", 9))

    def _begin_camera_scan(self) -> None:
        """Probe camera indices outside Tk's event loop."""

        if self._scanning:
            return
        if self._active_source.is_open:
            messagebox.showinfo("提示", "请先停止当前摄像头，再刷新设备列表。")
            return
        self._scanning = True
        self._set_motion_state("connecting")
        self.scan_button.configure(state="disabled")
        self.start_button.configure(state="disabled")
        self.camera_combo.configure(state="disabled")
        self._devices = {}
        self.camera_choice.set("正在扫描摄像头…")

        def worker() -> None:
            try:
                self._device_queue.put((self.camera.discover(), None))
            except Exception as error:  # Camera backends vary across platforms.
                self._device_queue.put((None, str(error)))

        threading.Thread(target=worker, daemon=True).start()

    def _consume_device_scan(self) -> None:
        """Apply a completed camera scan on the Tk thread."""

        try:
            devices, error = self._device_queue.get_nowait()
        except queue.Empty:
            return
        self._scanning = False
        self._set_motion_state("error" if error else "idle")
        self.scan_button.configure(state="normal")
        self.start_button.configure(state="disabled")
        self.camera_combo.configure(state="disabled", values=())
        self._devices = {}
        if error:
            self.camera_choice.set("扫描失败")
            self.footer_text.set(f"读取设备列表失败：{error}；可点击刷新重试")
            return
        assert devices is not None
        self._devices = {device.label: device for device in devices}
        if not devices:
            self.camera_combo.configure(values=())
            self.camera_choice.set("未找到电脑摄像头")
            self.footer_text.set("未找到可用的电脑摄像头；请检查设备连接后刷新，手机相机不会作为替代")
            return
        labels = list(self._devices)
        self.camera_combo.configure(values=labels)
        preferred = next(
            (device.label for device in devices if device.built_in),
            next((device.label for device in devices if device.index == self.config.camera.preferred_index), labels[0]),
        )
        self.camera_choice.set(preferred)
        self.camera_combo.configure(state="readonly")
        self.start_button.configure(state="normal")
        self.footer_text.set("已读取设备列表 · 点击开启后才启动镜头")

    def _toggle_camera(self) -> None:
        """Start or stop the selected camera."""

        if self._active_source.is_open:
            self._stop_camera()
            return
        device = self._devices.get(self.camera_choice.get())
        if device is None:
            messagebox.showwarning("未选择设备", "请先刷新并选择摄像头。")
            return
        try:
            self.camera.open(device.index, unique_id=device.unique_id)
        except RuntimeError as error:
            self._set_motion_state("error")
            messagebox.showerror("摄像头启动失败", str(error))
            self.footer_text.set(str(error))
            return
        self.voter.reset()
        self.continuity.reset()
        self.generation += 1
        self._last_analysis = 0.0
        self.camera_combo.configure(state="disabled")
        self._read_failures = 0
        self._source_requested = True
        self._set_motion_state("connecting")
        self.connection_text.set(f"等待首帧 · 设备 {self.camera.current_index}")
        self.start_button.configure(text="停止摄像头")
        self.footer_text.set("正在等待摄像头画面；请正对镜头并保持光线均匀")

    @property
    def _active_source(self):
        return self._video_source or self.camera

    def _open_video(self):
        path = filedialog.askopenfilename(title="选择本地测试视频", filetypes=[("视频", "*.mp4 *.mov *.avi *.mkv"), ("所有文件", "*")])
        if not path:
            return
        self._stop_camera()
        try:
            from .video import VideoFileSource
            self._video_source = VideoFileSource(path)
        except (OSError, ValueError, cv2.error) as error:
            self._set_motion_state("error")
            messagebox.showerror("视频打开失败", str(error))
            return
        self._last_analysis = 0.
        self._source_requested = True
        self._set_motion_state("connecting")
        self.frame_number = 0
        self.camera_combo.configure(state="disabled")
        self.start_button.configure(state="normal", text="停止视频")
        self.connection_text.set("本地视频 · 等待分析")
        self.footer_text.set(f"视频回放：{Path(path).name} · 未开启摄像头")

    def _stop_camera(self) -> None:
        """Stop capture and clear temporal recognition state."""

        self.camera.release()
        self._source_requested = False
        self._set_motion_state("idle")
        if self._video_source is not None:
            self._video_source.release()
            self._video_source = None
        self.voter.reset()
        self.continuity.reset()
        self.generation += 1
        self.camera_combo.configure(state="readonly")
        self.current_frame = None
        self.current_observations = []
        self._refresh_tracks([])
        self.last_stable_key = None
        self.connection_text.set("未连接")
        self.start_button.configure(text="开启摄像头")
        self.result_name.set("等待摄像头")
        self.result_detail.set("选择设备并启动实时比对")
        self.quality_text.set("质量指标将在检测后显示")
        self.preview.configure(image="", text="")
        self.empty_preview.place(relwidth=1, relheight=1)
        self.result_label.configure(foreground=self.TEXT)
        self.footer_text.set("摄像头已停止")

    def _tick(self) -> None:
        """Read, analyze, and render camera frames without blocking Tk."""

        self._consume_device_scan()
        if self._source_requested and not self._active_source.is_open:
            self._stop_camera()
            self.result_name.set("视频源连接已中断")
            self.result_detail.set("旧识别结果已清空，请重新开启相机或视频")
            self._set_motion_state("error")
        packet = self.worker.poll()
        if packet is not None:
            generation, captured, observations, error, elapsed = packet
            if generation == self.generation and self._active_source.is_open and time.monotonic()-captured < 1.0:
                self.current_observations = observations
                self._last_analysis = captured
                if error:
                    self._set_motion_state("error")
                    self._refresh_tracks([])
                    self.voter.reset()
                    self.continuity.reset()
                    self.result_name.set("识别暂停")
                    self.result_label.configure(foreground=self.WARNING)
                    self.result_detail.set(error)
                else:
                    self._update_result_panel(observations)
                    self.connection_text.set(f"● {self._active_source.current_index} · 分析 {elapsed:.0f} ms")
        if self._active_source.is_open:
            ok, frame = self._active_source.read()
            if ok is None:
                self._schedule_tick()
                return
            if ok and frame is not None:
                self._read_failures = 0
                if self._video_source is None:
                    frame = cv2.flip(frame, 1)
                self.current_frame = frame.copy()
                self._current_frame_time = time.monotonic()
                self.frame_number += 1
                if self.frame_number % max(1, self.config.ui.process_every_n_frames) == 0:
                    self.worker.submit(frame, self.generation, self._current_frame_time,
                                       tracking_time=self._video_source.timestamp if self._video_source else None)
                if time.monotonic()-self._last_analysis > 1.0:
                    self._set_motion_state("connecting")
                    self._refresh_tracks([])
                    self.current_observations = []
                    self.voter.reset()
                    self.continuity.reset()
                    self.result_name.set("等待新画面分析…")
                    self.result_label.configure(foreground=self.TEXT)
                    self.result_detail.set("当前尚无有效识别结果")
                rendered = self.system.annotate(frame, self.current_observations)
                self._show_frame(rendered)
            else:
                if self._video_source is not None:
                    self._stop_camera()
                    self.result_name.set("视频已结束或无法继续解码")
                    self.result_detail.set("可重新打开视频；离线逐帧结果可通过 video 命令导出")
                    self.after(30, self._tick)
                    return
                self._read_failures = getattr(self, "_read_failures", 0)+1
                if self.current_frame is not None:
                    self.generation += 1
                self.current_frame = None
                self.current_observations = []
                self._refresh_tracks([])
                self.voter.reset()
                self.continuity.reset()
                self.result_name.set("摄像头读取失败")
                self._set_motion_state("error")
                self.result_label.configure(foreground=self.WARNING)
                self.result_detail.set("恢复画面后将重新确认身份")
                self.preview.configure(image="", text="")
                self.empty_preview.place(relwidth=1, relheight=1)
                self.footer_text.set("读取摄像头画面失败，正在重试…")
                if self._read_failures >= 5:
                    self._stop_camera()
                    self.result_name.set("摄像头没有返回画面")
                    self.result_detail.set("请关闭占用相机的应用，检查系统相机权限后重新开启。")
                    self.footer_text.set("连接已停止 · 未切换其他设备；请确认运行程序的终端/Python具有相机权限")
        self._schedule_tick()

    def _schedule_tick(self):
        delay = self._video_source.poll_delay_ms() if self._video_source else 30
        self.after(delay, self._tick)

    def _update_result_panel(self, observations: list[FaceObservation]) -> None:
        """Show quality diagnostics and a stable multi-frame decision."""

        self.result_label.configure(foreground=self.TEXT)
        self._refresh_tracks(observations)
        if not observations:
            self._set_motion_state("searching" if self._source_requested else "idle")
            self.voter.reset()
            self.continuity.reset()
            self.last_stable_key = None
            self.result_name.set("未检测到人脸")
            self.result_detail.set("请进入画面中央，距离摄像头约 0.5–1.2 米")
            self.quality_text.set("等待有效人脸")
            return
        if len(observations) != 1:
            self.voter.reset()
            self.continuity.reset()
            confirmed = [o for o in observations if o.stable_recognition is not None]
            known = sum(o.stable_recognition.known for o in confirmed)
            if any(o.tracking_state == "identity_conflict" for o in observations):
                self._set_motion_state("error")
            elif any(o.tracking_state == "confirming" for o in observations):
                self._set_motion_state("confirming")
            else:
                self._set_motion_state("known" if known == len(observations) else "mixed")
            self.result_name.set(f"{len(observations)} 张人脸 · {known} 人已确认")
            lines = [f"T{o.track_id}：{o.stable_recognition.name if o.stable_recognition else self._track_status(o.tracking_state)}"
                     for o in observations[:3]]
            self.result_detail.set("  /  ".join(lines)+"\n完整结果见“视频轨迹”；录入时仅保留一人。")
            self.quality_text.set(f"当前 {len(observations)} 张人脸")
            return
        primary = max(observations, key=lambda item: item.box.area)
        report = primary.quality
        quality_state = "合格" if report.accepted else "不合格：" + "、".join(report.reasons)
        self.quality_text.set(
            f"画质 {quality_state}  ·  亮度 {report.brightness:.0f}  ·  "
            f"{report.focus_description}  ·  人脸占比 {report.face_size_ratio:.1%}"
        )
        if not report.accepted:
            self._set_motion_state("quality")
            self.voter.reset()
            self.continuity.reset()
            self.result_name.set("请调整画面")
            self.result_detail.set("、".join(report.reasons))
            return
        if primary.tracking_state != "untracked":
            if primary.stable_recognition is not None:
                self._display_stable_result(primary.stable_recognition)
            else:
                self._set_motion_state("confirming" if primary.tracking_state == "confirming" else "error")
                self.result_name.set(self._track_status(primary.tracking_state))
                self.result_detail.set(f"轨迹 T{primary.track_id} · 等待该人脸的独立多帧确认")
            return
        if not self.continuity.update(primary, time.monotonic()):
            self.voter.reset()
        if primary.recognition is None:
            return
        stable = self.voter.update(primary.recognition)
        if stable is None:
            self._set_motion_state("confirming")
            self.result_name.set("正在确认…")
            self.result_detail.set(
                f"单帧候选：{primary.recognition.name} · 匹配分 {primary.recognition.similarity:.1f}（非概率）"
            )
            return
        self._display_stable_result(stable)

    @staticmethod
    def _track_status(state):
        return {"known": "已确认", "unknown": "未知人员", "confirming": "正在确认…",
                "quality_rejected": "请调整画面", "identity_conflict": "身份冲突",
                "capacity": "超出跟踪容量"}.get(state, "待分析")

    def _refresh_tracks(self, observations):
        for row in self.track_list.get_children():
            self.track_list.delete(row)
        for index, observation in enumerate(observations):
            result = observation.stable_recognition
            self.track_list.insert("", "end", iid=str(index), values=(
                f"T{observation.track_id}" if observation.track_id else "—",
                result.name if result else "—", self._track_status(observation.tracking_state)))

    def _display_stable_result(self, result: RecognitionResult) -> None:
        """Render and log a stable temporal decision."""

        self._set_motion_state("known" if result.known else "unknown")
        self.result_name.set(result.name)
        self.result_label.configure(foreground=self.SUCCESS if result.known else self.DANGER)
        verdict = "比对通过" if result.known else "比对不通过"
        self.result_detail.set(
            f"{verdict} · 匹配分 {result.similarity:.1f}（非概率） · "
            f"距离 {result.distance:.3f} / 阈值 {result.threshold:.3f}\n{result.reason}"
        )
        key = (result.known, result.person_id)
        if key != self.last_stable_key:
            self.system.logger.write(
                "stable_recognition",
                known=result.known,
                name=result.name,
                similarity=round(result.similarity, 3),
                distance=round(result.distance, 6),
                threshold=round(result.threshold, 6),
                reason=result.reason,
            )
            if time.monotonic() >= self._feedback_until:
                self.footer_text.set(f"{datetime.now():%H:%M:%S} · {verdict}：{result.name}")
            self.last_stable_key = key

    def _show_frame(self, frame) -> None:
        """Scale a BGR frame into the preview area while preserving aspect ratio."""

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        width = max(self.preview.winfo_width(), 320)
        height = max(self.preview.winfo_height(), 240)
        image.thumbnail((width, height), Image.Resampling.LANCZOS)
        self._preview_photo = ImageTk.PhotoImage(image)
        self.empty_preview.place_forget()
        self.preview.configure(image=self._preview_photo, text="")

    def _capture_enrollment_sample(self) -> None:
        """Validate and stage one current camera frame for enrollment."""

        if self.current_frame is None or time.monotonic()-self._current_frame_time > 1.0:
            messagebox.showwarning("没有画面", "请先启动摄像头。")
            return
        try:
            self.session.validate_name(self.person_name.get())
            with self.system.lock:
                sample = self.system.prepare_sample(self.current_frame)
                self.session.add(self.person_name.get(), sample,
                                 extractor=self.system.extractor if self.config.engine.backend == "sface" else None,
                                 consistency=self.config.engine.enrollment_consistency)
        except (ValueError, RuntimeError, cv2.error) as error:
            messagebox.showwarning("样本未采集", str(error))
            self.footer_text.set(str(error))
            self._feedback("warning")
            return
        self.name_entry.configure(state="disabled")
        target = self.config.ui.enrollment_target_samples
        count = len(self.enrollment_samples)
        self._animate_sample_progress(min(count, target))
        self.sample_text.set(f"{count} / {target}")
        preview = Image.fromarray(cv2.cvtColor(sample.crop, cv2.COLOR_BGR2RGB))
        preview.thumbnail((52, 52))
        photo = ImageTk.PhotoImage(preview)
        self._sample_photos.append(photo)
        for widget in self.thumbnail_frame.winfo_children():
            widget.destroy()
        for item in self._sample_photos[-5:]:
            tk.Label(self.thumbnail_frame, image=item, bg=self.CARD).pack(side="left", padx=2)
        prompts = ["轻微向左转，停稳后采集", "轻微向右转，停稳后采集", "正脸自然微笑后采集", "恢复正脸，稍微改变距离后采集"]
        self.guide_text.set(prompts[count-1] if count <= 4 else "样本已齐，确认缩略图都是同一个人后点“保存入库”")
        self.footer_text.set(f"已采集第 {count} 张样本；请稍微改变角度或表情")
        self._feedback()

    def _save_enrollment(self) -> None:
        """Commit staged samples and calibrate the updated database."""

        name = self.person_name.get().strip()
        if not name:
            messagebox.showwarning("姓名为空", "请输入要录入的人员姓名。")
            return
        existing = any(
            person.name.casefold() == name.casefold()
            for person in self.system.database.list_people()
        )
        minimum = 1 if existing else min(3, self.config.ui.enrollment_target_samples)
        if len(self.enrollment_samples) < minimum:
            messagebox.showwarning(
                "样本不足",
                f"{'新增人员' if not existing else '追加样本'}至少需要 {minimum} 张，本次已采集 {len(self.enrollment_samples)} 张。",
            )
            return
        try:
            self.session.validate_name(name)
            with self.system.lock:
                person = self.system.enroll_samples(name, self.enrollment_samples)
        except (ValueError, OSError, RuntimeError, sqlite3.Error) as error:
            messagebox.showerror("保存失败", str(error))
            self.footer_text.set(f"保存失败：{error}")
            self._feedback("error")
            return
        added = len(self.enrollment_samples)
        self._clear_enrollment()
        self._refresh_database_panel()
        self.voter.reset()
        self.continuity.reset()
        self.generation += 1
        self.current_observations = []
        self._refresh_tracks([])
        self.result_name.set("标准库已更新")
        self.result_label.configure(foreground=self.TEXT)
        self.result_detail.set("等待新画面重新确认")
        self._set_motion_state("searching" if self._source_requested else "idle")
        self.footer_text.set(
            f"已为 {person.name} 保存 {added} 张样本（共 {person.sample_count} 张）；请用新画面测试"
        )
        self._feedback()  # Non-modal success feedback; text remains after animation.

    def _clear_enrollment(self):
        self.motion.cancel("samples")
        self.session.clear()
        self.name_entry.configure(state="normal")
        self.person_name.set("")
        self.sample_progress.configure(value=0)
        self.sample_text.set(f"0 / {self.config.ui.enrollment_target_samples}")
        self._sample_photos.clear()
        for widget in self.thumbnail_frame.winfo_children():
            widget.destroy()
        self.guide_text.set("第1步：输入姓名，正对镜头，画质合格后采集")

    def _manage_people(self):
        window = tk.Toplevel(self)
        window.title("标准库人员管理")
        window.geometry("580x440")
        window.minsize(530, 380)
        window.configure(bg=self.BACKGROUND)
        window.transient(self)
        ttk.Label(window, text="人员管理", style="Title.TLabel").pack(anchor="w", padx=20, pady=(20, 4))
        ttk.Label(window, text="选择姓名后删除；原照片与特征保留本地备份，并非永久擦除。", style="Subtitle.TLabel").pack(anchor="w", padx=20)
        listing = ttk.Treeview(window, columns=("name", "count"), show="headings", selectmode="browse")
        listing.heading("name", text="姓名")
        listing.heading("count", text="样本数")
        listing.pack(fill="both", expand=True, padx=12, pady=12)

        def refresh():
            for row in listing.get_children():
                listing.delete(row)
            for person in self.system.database.list_people():
                listing.insert("", "end", iid=person.person_id, values=(person.name, person.sample_count))
            delete_button.configure(state="disabled")
            delete_all_button.configure(state="normal" if listing.get_children() else "disabled")

        def remove(all_people=False):
            selection = listing.selection()
            if not all_people and not selection:
                return
            if self._delete_saved_people(None if all_people else selection[0], parent=window):
                refresh()

        delete_button = ttk.Button(window, text="删除所选人员", style="Danger.TButton", command=remove)
        delete_button.pack(side="left", padx=12, pady=12)
        delete_all_button = ttk.Button(window, text="删除全部人员（保留备份）", style="Ghost.TButton", command=lambda: remove(True))
        delete_all_button.pack(side="right", padx=12, pady=12)
        listing.bind("<<TreeviewSelect>>", lambda _event: delete_button.configure(state="normal" if listing.selection() else "disabled"))
        refresh()

    def _calibrate(self) -> None:
        """Run explicit database threshold calibration."""

        try:
            with self.system.lock:
                result = self.system.calibrate_threshold()
        except Exception as error:
            messagebox.showerror("校准失败", str(error))
            return
        self._refresh_database_panel()
        rates = ""
        if result.false_accept_rate is not None:
            rates = f"\n库内估计 FAR {result.false_accept_rate:.1%}，FRR {result.false_reject_rate:.1%}"
        messagebox.showinfo(
            "阈值校准完成",
            f"阈值：{result.threshold:.3f}\n来源：{result.source}{rates}",
        )

    def _refresh_database_panel(self) -> None:
        """Refresh identity/sample counts and threshold provenance."""

        people = self.system.database.list_people()
        self.database_text.set(f"{len(people)} 人 · {sum(item.sample_count for item in people)} 个样本")
        selected = self.saved_people_list.selection()
        for row in self.saved_people_list.get_children():
            self.saved_people_list.delete(row)
        for person in people:
            self.saved_people_list.insert("", "end", iid=person.person_id, values=(person.name, person.sample_count))
        if selected and self.saved_people_list.exists(selected[0]):
            self.saved_people_list.selection_set(selected[0])
        self.saved_hint.set("点击姓名，再点上方红色删除按钮。" if people else "还没有已保存的人员。请先到“录入人员”采集并保存。")
        self._update_saved_selection()
        calibrated = self.system.database.calibrated_threshold
        if calibrated is None:
            self.threshold_text.set(
                f"距离阈值 {self.system.recognizer.threshold:.3f}（{self.config.engine.backend}）"
            )
        else:
            self.threshold_text.set(f"判定阈值 {calibrated:.3f}（自动校准）")

    def _on_close(self) -> None:
        """Release camera hardware before destroying the window."""

        self.motion.close()
        self.camera.release()
        if self._video_source is not None:
            self._video_source.release()
        self.worker.close()
        self.worker.thread.join(timeout=.5)
        if not self.worker.thread.is_alive() and hasattr(self.system.database, "close"):
            self.system.database.close()
        self.destroy()


def launch_ui(config: AppConfig, project_root: str | Path) -> None:
    """Construct and run the desktop application."""

    app = FaceCompareApp(config, project_root)
    app.mainloop()
