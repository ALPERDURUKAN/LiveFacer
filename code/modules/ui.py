import os
import webbrowser
import customtkinter as ctk
from typing import Callable, Tuple, Any, List, Optional
import queue
import threading
import cv2
from cv2_enumerate_cameras import enumerate_cameras  # Add this import
from PIL import Image, ImageOps
import time
import json

import modules.globals
import modules.metadata
import modules.virtual_camera
from modules.ui_tooltip import ToolTip
from tkinter import messagebox
from modules.gettext import LanguageManager
from modules.face_analyser import (
    get_one_face,
    get_unique_faces_from_target_image,
    get_unique_faces_from_target_video,
    add_blank_map,
    has_valid_map,
    simplify_maps,
)
from modules.capturer import get_video_frame, get_video_frame_total
from modules.processors.frame.core import get_frame_processors_modules
from modules.video_capture import VideoCapturer
from modules.utilities import (
    is_image,
    is_video,
    resolve_relative_path,
    has_image_extension,
    read_static_image,
)

ROOT = None
POPUP = None
POPUP_LIVE = None
ROOT_HEIGHT = 680
ROOT_WIDTH = 640

PREVIEW = None
PREVIEW_MAX_HEIGHT = 700
PREVIEW_MAX_WIDTH = 1200
PREVIEW_DEFAULT_WIDTH = 960
PREVIEW_DEFAULT_HEIGHT = 540

POPUP_WIDTH = 750
POPUP_HEIGHT = 810
POPUP_SCROLL_WIDTH = (740,)
POPUP_SCROLL_HEIGHT = 700

POPUP_LIVE_WIDTH = 900
POPUP_LIVE_HEIGHT = 820
POPUP_LIVE_SCROLL_WIDTH = (890,)
POPUP_LIVE_SCROLL_HEIGHT = 700

MAPPER_PREVIEW_MAX_HEIGHT = 100
MAPPER_PREVIEW_MAX_WIDTH = 100
FACE_PREVIEW_SIZE = 180

DEFAULT_BUTTON_WIDTH = 200
DEFAULT_BUTTON_HEIGHT = 34

RECENT_DIRECTORY_SOURCE = None
RECENT_DIRECTORY_TARGET = None
RECENT_DIRECTORY_OUTPUT = None

preview_label = None
preview_slider = None
source_label = None
target_label = None
status_label = None
popup_status_label = None
popup_status_label_live = None
source_label_dict = {}
source_label_dict_live = {}
target_label_dict_live = {}

_ = None  # Translation function

img_ft, vid_ft = modules.globals.file_types
DETECT_EVERY_N = 2


def tr(text: str) -> str:
    if callable(_):
        try:
            return _(text)
        except Exception:
            return text
    return text


def prompt_license_activation(initial_message: str = "") -> Optional[str]:
    temp_root = ctk.CTk()
    temp_root.withdraw()
    try:
        if initial_message:
            messagebox.showinfo(
                title=tr("License Activation"),
                message=tr(initial_message),
                parent=temp_root,
            )
        dialog = ctk.CTkInputDialog(
            text=tr("Enter your license key"),
            title=tr("License Activation"),
        )
        value = dialog.get_input()
        if value is None:
            return None
        stripped = value.strip()
        return stripped if stripped else None
    finally:
        try:
            temp_root.destroy()
        except Exception:
            pass


def init(start: Callable[[], None], destroy: Callable[[], None], lang: str = "en") -> ctk.CTk:
    global ROOT, PREVIEW, _

    lang_manager = LanguageManager(lang)
    _ = lang_manager._

    ROOT = create_root(start, destroy)
    PREVIEW = create_preview(ROOT)

    return ROOT


def save_switch_states():
    switch_states = {
        "keep_fps": modules.globals.keep_fps,
        "keep_audio": modules.globals.keep_audio,
        "keep_frames": modules.globals.keep_frames,
        "many_faces": modules.globals.many_faces,
        "map_faces": modules.globals.map_faces,
        "poisson_blend": modules.globals.poisson_blend,
        "color_correction": modules.globals.color_correction,
        "nsfw_filter": modules.globals.nsfw_filter,
        "live_mirror": modules.globals.live_mirror,
        "live_resizable": modules.globals.live_resizable,
        "live_width": modules.globals.live_width,
        "live_height": modules.globals.live_height,
        "fp_ui": modules.globals.fp_ui,
        "show_fps": modules.globals.show_fps,
        "mouth_mask": modules.globals.mouth_mask,
        "show_mouth_mask_box": modules.globals.show_mouth_mask_box,
        "eyes_mask": modules.globals.eyes_mask,
        "show_eyes_mask_box": modules.globals.show_eyes_mask_box,
        "eyebrows_mask": modules.globals.eyebrows_mask,
        "show_eyebrows_mask_box": modules.globals.show_eyebrows_mask_box,
        "virtual_camera": modules.globals.virtual_camera,
    }
    with open("switch_states.json", "w") as f:
        json.dump(switch_states, f)


def load_switch_states():
    try:
        with open("switch_states.json", "r") as f:
            switch_states = json.load(f)
        modules.globals.keep_fps = switch_states.get("keep_fps", True)
        modules.globals.keep_audio = switch_states.get("keep_audio", True)
        modules.globals.keep_frames = switch_states.get("keep_frames", False)
        modules.globals.many_faces = switch_states.get("many_faces", False)
        modules.globals.map_faces = switch_states.get("map_faces", False)
        modules.globals.poisson_blend = switch_states.get("poisson_blend", False)
        modules.globals.color_correction = switch_states.get("color_correction", False)
        modules.globals.nsfw_filter = switch_states.get("nsfw_filter", False)
        modules.globals.live_mirror = switch_states.get("live_mirror", False)
        modules.globals.live_resizable = switch_states.get("live_resizable", False)
        modules.globals.live_width = switch_states.get("live_width", modules.globals.live_width)
        modules.globals.live_height = switch_states.get("live_height", modules.globals.live_height)
        modules.globals.fp_ui = switch_states.get("fp_ui", {"face_enhancer": False})
        modules.globals.show_fps = switch_states.get("show_fps", False)
        modules.globals.mouth_mask = switch_states.get("mouth_mask", False)
        modules.globals.show_mouth_mask_box = switch_states.get("show_mouth_mask_box", False)
        modules.globals.eyes_mask = switch_states.get("eyes_mask", False)
        modules.globals.show_eyes_mask_box = switch_states.get("show_eyes_mask_box", False)
        modules.globals.eyebrows_mask = switch_states.get("eyebrows_mask", False)
        modules.globals.show_eyebrows_mask_box = switch_states.get("show_eyebrows_mask_box", False)
        modules.globals.virtual_camera = switch_states.get("virtual_camera", False)
    except FileNotFoundError:
        # If the file doesn't exist, use default values
        pass


def create_root(start: Callable[[], None], destroy: Callable[[], None]) -> ctk.CTk:
    global source_label, target_label, status_label, show_fps_switch

    load_switch_states()

    ctk.deactivate_automatic_dpi_awareness()
    ctk.set_appearance_mode("system")
    ctk.set_default_color_theme(resolve_relative_path("ui.json"))

    root = ctk.CTk()
    root.minsize(ROOT_WIDTH, ROOT_HEIGHT)
    root.title(
        f"{modules.metadata.name} {modules.metadata.version} {modules.metadata.edition}"
    )
    root.configure()
    root.protocol("WM_DELETE_WINDOW", lambda: destroy())
    root.grid_rowconfigure(0, weight=1)
    root.grid_columnconfigure(0, weight=1)

    content = ctk.CTkFrame(root, fg_color="transparent")
    content.grid(row=0, column=0, sticky="nsew", padx=16, pady=(8, 6))
    content.grid_columnconfigure(0, weight=1)
    content.grid_rowconfigure(2, weight=1)

    face_preview_frame = ctk.CTkFrame(content, fg_color="transparent")
    face_preview_frame.grid(row=0, column=0, sticky="ew", pady=(0, 4))
    face_preview_frame.grid_columnconfigure(0, weight=1)
    face_preview_frame.grid_columnconfigure(1, weight=0)
    face_preview_frame.grid_columnconfigure(2, weight=1)

    # Enhanced image preview containers with glassmorphism
    source_container = ctk.CTkFrame(face_preview_frame, corner_radius=16, border_width=1)
    source_container.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
    source_container.grid_columnconfigure(0, weight=1)
    source_container.grid_rowconfigure(0, weight=1)

    source_label = ctk.CTkLabel(
        source_container,
        text=None,
        width=FACE_PREVIEW_SIZE,
        height=FACE_PREVIEW_SIZE,
        corner_radius=12,
        fg_color="transparent"
    )
    source_label.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

    swap_icon = ctk.CTkLabel(
        face_preview_frame,
        text="↔",
        font=ctk.CTkFont(size=28, weight="bold"),
        text_color="#FFB84D"
    )
    swap_icon.grid(row=0, column=1, padx=8)

    target_container = ctk.CTkFrame(face_preview_frame, corner_radius=16, border_width=1)
    target_container.grid(row=0, column=2, sticky="nsew", padx=(16, 0))
    target_container.grid_columnconfigure(0, weight=1)
    target_container.grid_rowconfigure(0, weight=1)

    target_label = ctk.CTkLabel(
        target_container,
        text=None,
        width=FACE_PREVIEW_SIZE,
        height=FACE_PREVIEW_SIZE,
        corner_radius=12,
        fg_color="transparent"
    )
    target_label.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

    button_frame = ctk.CTkFrame(content, fg_color="transparent")
    button_frame.grid(row=1, column=0, sticky="ew", pady=(6, 8))
    button_frame.grid_columnconfigure((0, 1, 2), weight=1, uniform="face-buttons")

    select_face_button = ctk.CTkButton(
        button_frame,
        text=tr("Select a face"),
        cursor="hand2",
        command=select_source_path,
        height=DEFAULT_BUTTON_HEIGHT,
        font=ctk.CTkFont(size=14, weight="normal")
    )
    select_face_button.grid(row=0, column=0, sticky="ew", padx=(0, 12))
    ToolTip(select_face_button, tr("Choose the source face image to swap onto the target"))

    swap_faces_button = ctk.CTkButton(
        button_frame,
        text="↔",
        cursor="hand2",
        command=swap_faces_paths,
        width=52,
        height=DEFAULT_BUTTON_HEIGHT,
        font=ctk.CTkFont(size=18, weight="bold")
    )
    swap_faces_button.grid(row=0, column=1, sticky="ew", padx=12)
    ToolTip(swap_faces_button, tr("Swap source and target images"))

    select_target_button = ctk.CTkButton(
        button_frame,
        text=tr("Select a target"),
        cursor="hand2",
        command=select_target_path,
        height=DEFAULT_BUTTON_HEIGHT,
        font=ctk.CTkFont(size=14, weight="normal")
    )
    select_target_button.grid(row=0, column=2, sticky="ew", padx=(12, 0))
    ToolTip(select_target_button, tr("Choose the target image or video to apply face swap to"))

    switch_container = ctk.CTkFrame(content, corner_radius=16)
    switch_container.grid(row=2, column=0, sticky="nsew", pady=(0, 12))
    switch_container.grid_columnconfigure(0, weight=1)
    switch_container.grid_rowconfigure(0, weight=1)

    SWITCH_SCROLL_HEIGHT = 150
    switch_frame = ctk.CTkScrollableFrame(
        switch_container,
        height=SWITCH_SCROLL_HEIGHT
    )
    switch_frame.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
    switch_frame.grid_columnconfigure((0, 1, 2), weight=1, uniform="switch-columns")

    def set_attr(attr: str, value: bool):
        setattr(modules.globals, attr, value)

    def set_map_faces(value: bool):
        modules.globals.map_faces = value
        if not value:
            close_mapper_window()

    switch_specs = [
        {"text": tr("Eyebrows Mask"), "getter": lambda: modules.globals.eyebrows_mask, "setter": lambda val: set_attr("eyebrows_mask", val), "tooltip": None},
        {"text": tr("Eyes Mask"), "getter": lambda: modules.globals.eyes_mask, "setter": lambda val: set_attr("eyes_mask", val), "tooltip": None},
        {"text": tr("Mouth Mask"), "getter": lambda: modules.globals.mouth_mask, "setter": lambda val: set_attr("mouth_mask", val), "tooltip": None},
        {"text": tr("Keep fps"), "getter": lambda: modules.globals.keep_fps, "setter": lambda val: set_attr("keep_fps", val), "tooltip": tr("Output video keeps the original frame rate")},
        {"text": tr("Keep frames"), "getter": lambda: modules.globals.keep_frames, "setter": lambda val: set_attr("keep_frames", val), "tooltip": tr("Keep extracted frames on disk after processing")},
        {"text": tr("Face Enhancer"), "getter": lambda: modules.globals.fp_ui["face_enhancer"], "setter": lambda val: update_tumbler("face_enhancer", val), "tooltip": tr("Select a face enhancement model (None = no enhancement)")},
        {"text": tr("Map faces"), "getter": lambda: modules.globals.map_faces, "setter": lambda val: set_map_faces(val), "tooltip": tr("Manually assign which source face maps to which target face")},
        {"text": tr("Poisson Blend"), "getter": lambda: modules.globals.poisson_blend, "setter": lambda val: set_attr("poisson_blend", val), "tooltip": tr("Blend face edges smoothly using Poisson blending")},
        {"text": tr("Show Eyebrows Mask Box"), "getter": lambda: modules.globals.show_eyebrows_mask_box, "setter": lambda val: set_attr("show_eyebrows_mask_box", val), "tooltip": None},
        {"text": tr("Show Eyes Mask Box"), "getter": lambda: modules.globals.show_eyes_mask_box, "setter": lambda val: set_attr("show_eyes_mask_box", val), "tooltip": None},
        {"text": tr("Show Mouth Mask Box"), "getter": lambda: modules.globals.show_mouth_mask_box, "setter": lambda val: set_attr("show_mouth_mask_box", val), "tooltip": None},
        {"text": tr("Keep audio"), "getter": lambda: modules.globals.keep_audio, "setter": lambda val: set_attr("keep_audio", val), "tooltip": tr("Copy audio track from the source video to output")},
        {"text": tr("Many faces"), "getter": lambda: modules.globals.many_faces, "setter": lambda val: set_attr("many_faces", val), "tooltip": tr("Swap every detected face, not just the primary one")},
        {"text": tr("Fix Blueish Cam"), "getter": lambda: modules.globals.color_correction, "setter": lambda val: set_attr("color_correction", val), "tooltip": tr("Fix blue/green color cast from some webcams")},
        {"text": tr("Show FPS"), "getter": lambda: modules.globals.show_fps, "setter": lambda val: set_attr("show_fps", val), "tooltip": tr("Display frames-per-second counter on the live preview")},
        {"text": tr("Virtual Camera"), "getter": lambda: modules.globals.virtual_camera, "setter": lambda val: set_attr("virtual_camera", val), "tooltip": None},
    ]

    columns = 3
    rows_per_column = (len(switch_specs) + columns - 1) // columns
    switch_vars = []

    for idx, spec in enumerate(switch_specs):
        var = ctk.BooleanVar(value=spec["getter"]())
        switch_vars.append(var)

        def on_toggle(spec=spec, var=var):
            spec["setter"](var.get())
            save_switch_states()

        column = idx // rows_per_column
        row = idx % rows_per_column
        padx = (24, 12) if column == 0 else (12, 12) if column == 1 else (12, 24)

        switch = ctk.CTkSwitch(
            switch_frame,
            text=spec["text"],
            variable=var,
            cursor="hand2",
            command=on_toggle,
        )
        switch.grid(row=row, column=column, sticky="w", padx=padx, pady=8)
        if spec.get("tooltip"):
            ToolTip(switch, spec["tooltip"])

    switch_frame.switch_vars = switch_vars

    slider_frame = ctk.CTkFrame(content, fg_color="transparent")
    slider_frame.grid(row=3, column=0, sticky="ew", pady=(0, 10))
    slider_frame.grid_columnconfigure((0, 1), weight=1, uniform="sliders")

    sharpness_container = ctk.CTkFrame(slider_frame, fg_color="transparent")
    sharpness_container.grid(row=0, column=0, sticky="ew", padx=(0, 16))
    sharpness_container.grid_columnconfigure(0, weight=1)

    sharpness_var = ctk.DoubleVar(value=modules.globals.sharpness)

    def on_sharpness_change(value: float):
        val = float(value)
        modules.globals.sharpness = val
        percentage = int(val * 100)
        sharpness_label.configure(text=f"{tr('Sharpness')}: {percentage}%")

    sharpness_label = ctk.CTkLabel(
        sharpness_container,
        text=f"{tr('Sharpness')}: {int(modules.globals.sharpness * 100)}%",
        font=ctk.CTkFont(size=13, weight="normal")
    )
    sharpness_label.grid(row=0, column=0, sticky="w", pady=(0, 2))

    sharpness_slider = ctk.CTkSlider(
        sharpness_container,
        from_=0.0,
        to=1.0,
        variable=sharpness_var,
        command=on_sharpness_change,
    )
    sharpness_slider.grid(row=1, column=0, sticky="ew", pady=(4, 0))
    ToolTip(sharpness_slider, tr("Sharpen the enhanced face output"))

    opacity_container = ctk.CTkFrame(slider_frame, fg_color="transparent")
    opacity_container.grid(row=0, column=1, sticky="ew", padx=(16, 0))
    opacity_container.grid_columnconfigure(0, weight=1)

    opacity_var = ctk.DoubleVar(value=modules.globals.opacity)

    def on_opacity_change(value: float):
        val = float(value)
        modules.globals.opacity = val
        percentage = int(val * 100)
        opacity_label.configure(text=f"{tr('Similarity')}: {percentage}%")

    opacity_label = ctk.CTkLabel(
        opacity_container,
        text=f"{tr('Similarity')}: {int(modules.globals.opacity * 100)}%",
        font=ctk.CTkFont(size=13, weight="normal")
    )
    opacity_label.grid(row=0, column=0, sticky="w", pady=(0, 2))

    opacity_slider = ctk.CTkSlider(
        opacity_container,
        from_=0.0,
        to=1.0,
        variable=opacity_var,
        command=on_opacity_change,
    )
    opacity_slider.grid(row=1, column=0, sticky="ew", pady=(4, 0))
    ToolTip(opacity_slider, tr("Blend between original and swapped face (0% = original, 100% = fully swapped)"))

    enhancer_intensity_container = ctk.CTkFrame(slider_frame, fg_color="transparent")
    enhancer_intensity_container.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(24, 0))
    enhancer_intensity_container.grid_columnconfigure(0, weight=1)

    enhancer_intensity_var = ctk.DoubleVar(
        value=modules.globals.face_enhancer_intensity
    )

    def on_enhancer_intensity_change(value: float):
        val = float(value)
        val = max(0.0, min(1.0, val))
        modules.globals.face_enhancer_intensity = val
        percentage = int(val * 100)
        enhancer_intensity_label.configure(
            text=f"{tr('Enhancer Intensity')}: {percentage}%"
        )

    enhancer_intensity_label = ctk.CTkLabel(
        enhancer_intensity_container,
        text=(
            f"{tr('Enhancer Intensity')}: {int(modules.globals.face_enhancer_intensity * 100)}%"
        ),
        font=ctk.CTkFont(size=13, weight="normal"),
    )
    enhancer_intensity_label.grid(row=0, column=0, sticky="w", pady=(0, 2))

    enhancer_intensity_slider = ctk.CTkSlider(
        enhancer_intensity_container,
        from_=0.0,
        to=1.0,
        variable=enhancer_intensity_var,
        command=on_enhancer_intensity_change,
    )
    enhancer_intensity_slider.grid(row=1, column=0, sticky="ew", pady=(4, 0))

    camera_frame = ctk.CTkFrame(content, fg_color="transparent")
    camera_frame.grid(row=4, column=0, sticky="ew", pady=(0, 10))
    camera_frame.grid_columnconfigure(1, weight=1)
    camera_frame.grid_columnconfigure(4, weight=1)

    camera_label = ctk.CTkLabel(
        camera_frame,
        text=tr("Select Camera:"),
        font=ctk.CTkFont(size=13, weight="normal")
    )
    camera_label.grid(row=0, column=0, sticky="w", padx=(0, 16))

    available_camera_indices, available_camera_strings = get_available_cameras()
    camera_values = (
        available_camera_strings if available_camera_strings else [tr("No cameras found")]
    )
    camera_variable = ctk.StringVar(value=camera_values[0])
    camera_optionmenu = ctk.CTkOptionMenu(
        camera_frame, variable=camera_variable, values=camera_values, width=200
    )
    camera_optionmenu.grid(row=0, column=1, sticky="ew", padx=(0, 16))

    def on_live_click():
        if not available_camera_strings:
            update_status("No cameras detected.")
            return
        selected = camera_variable.get()
        try:
            camera_index = available_camera_strings.index(selected)
        except ValueError:
            update_status("Select a valid camera.")
            return
        webcam_preview(root, available_camera_indices[camera_index])

    live_button = ctk.CTkButton(
        camera_frame,
        text=tr("Live"),
        cursor="hand2",
        state="normal" if available_camera_strings else "disabled",
        height=DEFAULT_BUTTON_HEIGHT,
        command=on_live_click,
    )
    live_button.grid(row=0, column=2, padx=(16, 0))
    ToolTip(live_button, tr("Start real-time face swap using webcam"))

    RESOLUTION_MIN = 320
    RESOLUTION_MAX = 3840

    def normalize_resolution(entry_widget: ctk.CTkEntry, attribute: str):
        value = entry_widget.get().strip()
        fallback = getattr(modules.globals, attribute)
        try:
            numeric_value = int(value)
        except (TypeError, ValueError):
            numeric_value = fallback
        else:
            numeric_value = max(RESOLUTION_MIN, min(RESOLUTION_MAX, numeric_value))

        setattr(modules.globals, attribute, numeric_value)
        entry_widget.delete(0, "end")
        entry_widget.insert(0, str(numeric_value))
        save_switch_states()

    resolution_label = ctk.CTkLabel(
        camera_frame,
        text=tr("Live Resolution (px):"),
        font=ctk.CTkFont(size=13, weight="normal")
    )
    resolution_label.grid(row=1, column=0, sticky="w", pady=(6, 0))

    width_label = ctk.CTkLabel(
        camera_frame,
        text="W",
        font=ctk.CTkFont(size=12, weight="normal")
    )
    width_label.grid(row=1, column=1, sticky="w", padx=(0, 4), pady=(6, 0))

    width_entry = ctk.CTkEntry(
        camera_frame,
        width=72
    )
    width_entry.insert(0, str(modules.globals.live_width))
    width_entry.grid(row=1, column=2, sticky="w", padx=(0, 12), pady=(6, 0))
    width_entry.bind("<Return>", lambda event: normalize_resolution(width_entry, "live_width"))
    width_entry.bind("<FocusOut>", lambda event: normalize_resolution(width_entry, "live_width"))

    height_label = ctk.CTkLabel(
        camera_frame,
        text="H",
        font=ctk.CTkFont(size=12, weight="normal")
    )
    height_label.grid(row=1, column=3, sticky="w", padx=(0, 4), pady=(6, 0))

    height_entry = ctk.CTkEntry(
        camera_frame,
        width=72
    )
    height_entry.insert(0, str(modules.globals.live_height))
    height_entry.grid(row=1, column=4, sticky="w", pady=(6, 0))
    height_entry.bind("<Return>", lambda event: normalize_resolution(height_entry, "live_height"))
    height_entry.bind("<FocusOut>", lambda event: normalize_resolution(height_entry, "live_height"))

    action_frame = ctk.CTkFrame(content, fg_color="transparent")
    action_frame.grid(row=5, column=0, sticky="ew", pady=(8, 2))
    action_frame.grid_columnconfigure((0, 1, 2), weight=1, uniform="action-buttons")

    start_button = ctk.CTkButton(
        action_frame,
        text=tr("Start"),
        cursor="hand2",
        command=lambda: analyze_target(start, root),
        height=DEFAULT_BUTTON_HEIGHT,
        font=ctk.CTkFont(size=15, weight="bold"),
        fg_color="#6366F1",
        hover_color="#8B5CF6"
    )
    start_button.grid(row=0, column=0, sticky="ew", padx=(0, 12))
    ToolTip(start_button, tr("Begin processing the target image/video with selected face"))

    stop_button = ctk.CTkButton(
        action_frame,
        text=tr("Destroy"),
        cursor="hand2",
        command=destroy,
        height=DEFAULT_BUTTON_HEIGHT,
        font=ctk.CTkFont(size=14, weight="normal"),
        fg_color="#4A4D50",
        hover_color="#6E7174"
    )
    stop_button.grid(row=0, column=1, sticky="ew", padx=12)
    ToolTip(stop_button, tr("Stop processing and close the application"))

    preview_button = ctk.CTkButton(
        action_frame,
        text=tr("Preview"),
        cursor="hand2",
        command=toggle_preview,
        height=DEFAULT_BUTTON_HEIGHT,
        font=ctk.CTkFont(size=14, weight="normal"),
        fg_color="#6366F1",
        hover_color="#8B5CF6"
    )
    preview_button.grid(row=0, column=2, sticky="ew", padx=(12, 0))
    ToolTip(preview_button, tr("Show/hide a preview of the processed output"))

    footer_frame = ctk.CTkFrame(content, fg_color="transparent")
    footer_frame.grid(row=6, column=0, sticky="ew", pady=(4, 4))
    footer_frame.grid_columnconfigure(0, weight=1)

    donate_label = ctk.CTkLabel(
        footer_frame,
        text="LiveFacer",
        justify="center",
        cursor="hand2",
        font=ctk.CTkFont(size=16, weight="normal")
    )
    donate_label.pack(fill="x")
    donate_label.configure(
        text_color=ctk.ThemeManager.theme.get("URL").get("text_color")
    )
    donate_label.bind(
        "<Button>", lambda event: webbrowser.open(modules.metadata.website)
    )

    status_label = ctk.CTkLabel(
        footer_frame,
        text=None,
        justify="center",
        font=ctk.CTkFont(size=14, weight="normal")
    )
    status_label.pack(fill="x", pady=(4, 0))

    return root


def analyze_target(start: Callable[[], None], root: ctk.CTk):
    if POPUP != None and POPUP.winfo_exists():
        update_status("Please complete pop-up or close it.")
        return

    if modules.globals.map_faces:
        modules.globals.souce_target_map = []

        if is_image(modules.globals.target_path):
            update_status("Getting unique faces")
            get_unique_faces_from_target_image()
        elif is_video(modules.globals.target_path):
            update_status("Getting unique faces")
            get_unique_faces_from_target_video()

        if len(modules.globals.souce_target_map) > 0:
            create_source_target_popup(start, root, modules.globals.souce_target_map)
        else:
            update_status("No faces found in target")
    else:
        select_output_path(start)


def create_source_target_popup(
    start: Callable[[], None], root: ctk.CTk, map: list
) -> None:
    global POPUP, popup_status_label

    POPUP = ctk.CTkToplevel(root)
    POPUP.title(tr("Source x Target Mapper"))
    POPUP.geometry(f"{POPUP_WIDTH}x{POPUP_HEIGHT}")
    POPUP.focus()

    def on_submit_click(start):
        if has_valid_map():
            POPUP.destroy()
            select_output_path(start)
        else:
            update_pop_status("At least 1 source with target is required!")

    scrollable_frame = ctk.CTkScrollableFrame(
        POPUP, width=POPUP_SCROLL_WIDTH, height=POPUP_SCROLL_HEIGHT
    )
    scrollable_frame.grid(row=0, column=0, padx=16, pady=16, sticky="nsew")

    def on_button_click(map, button_num):
        map = update_popup_source(scrollable_frame, map, button_num)

    for item in map:
        id = item["id"]

        button = ctk.CTkButton(
            scrollable_frame,
            text=tr("Select source image"),
            command=lambda id=id: on_button_click(map, id),
            width=DEFAULT_BUTTON_WIDTH,
            height=DEFAULT_BUTTON_HEIGHT,
            font=ctk.CTkFont(size=14, weight="normal")
        )
        button.grid(row=id, column=0, padx=50, pady=12)

        x_label = ctk.CTkLabel(
            scrollable_frame,
            text=f"X",
            width=MAPPER_PREVIEW_MAX_WIDTH,
            height=MAPPER_PREVIEW_MAX_HEIGHT,
        )
        x_label.grid(row=id, column=2, padx=10, pady=10)

        image = Image.fromarray(cv2.cvtColor(item["target"]["cv2"], cv2.COLOR_BGR2RGB))
        image = image.resize(
            (MAPPER_PREVIEW_MAX_WIDTH, MAPPER_PREVIEW_MAX_HEIGHT), Image.LANCZOS
        )
        tk_image = ctk.CTkImage(image, size=image.size)

        target_image = ctk.CTkLabel(
            scrollable_frame,
            text=f"T-{id}",
            width=MAPPER_PREVIEW_MAX_WIDTH,
            height=MAPPER_PREVIEW_MAX_HEIGHT,
        )
        target_image.grid(row=id, column=3, padx=10, pady=10)
        target_image.configure(image=tk_image)

    popup_status_label = ctk.CTkLabel(
        POPUP,
        text=None,
        justify="center",
        font=ctk.CTkFont(size=14, weight="normal")
    )
    popup_status_label.grid(row=1, column=0, pady=16)

    close_button = ctk.CTkButton(
        POPUP,
        text=tr("Submit"),
        command=lambda: on_submit_click(start),
        font=ctk.CTkFont(size=15, weight="bold"),
        fg_color="#6366F1",
        hover_color="#8B5CF6"
    )
    close_button.grid(row=2, column=0, pady=16)


def update_popup_source(
    scrollable_frame: ctk.CTkScrollableFrame, map: list, button_num: int
) -> list:
    global source_label_dict

    source_path = ctk.filedialog.askopenfilename(
        title=tr("select an source image"),
        initialdir=RECENT_DIRECTORY_SOURCE,
        filetypes=[img_ft],
    )

    if "source" in map[button_num]:
        map[button_num].pop("source")
        source_label_dict[button_num].destroy()
        del source_label_dict[button_num]

    if source_path == "":
        return map
    else:
        cv2_img = read_static_image(source_path)
        face = get_one_face(cv2_img)

        if face:
            x_min, y_min, x_max, y_max = face["bbox"]

            map[button_num]["source"] = {
                "cv2": cv2_img[int(y_min) : int(y_max), int(x_min) : int(x_max)],
                "face": face,
            }

            image = Image.fromarray(
                cv2.cvtColor(map[button_num]["source"]["cv2"], cv2.COLOR_BGR2RGB)
            )
            image = image.resize(
                (MAPPER_PREVIEW_MAX_WIDTH, MAPPER_PREVIEW_MAX_HEIGHT), Image.LANCZOS
            )
            tk_image = ctk.CTkImage(image, size=image.size)

            source_image = ctk.CTkLabel(
                scrollable_frame,
                text=f"S-{button_num}",
                width=MAPPER_PREVIEW_MAX_WIDTH,
                height=MAPPER_PREVIEW_MAX_HEIGHT,
            )
            source_image.grid(row=button_num, column=1, padx=10, pady=10)
            source_image.configure(image=tk_image)
            source_label_dict[button_num] = source_image
        else:
            update_pop_status("Face could not be detected in last upload!")
        return map


def create_preview(parent: ctk.CTkToplevel) -> ctk.CTkToplevel:
    global preview_label, preview_slider

    preview = ctk.CTkToplevel(parent)
    preview.withdraw()
    preview.title(tr("Preview"))
    preview.configure()
    preview.protocol("WM_DELETE_WINDOW", lambda: toggle_preview())
    preview.resizable(width=True, height=True)

    preview_label = ctk.CTkLabel(preview, text=None)
    preview_label.pack(fill="both", expand=True)

    preview_slider = ctk.CTkSlider(
        preview, from_=0, to=0, command=lambda frame_value: update_preview(frame_value)
    )

    return preview


def update_status(text: str) -> None:
    if status_label is not None:
        status_label.configure(text=tr(text))
    if ROOT is not None:
        ROOT.update()


def close_mapper_window() -> None:
    global POPUP, POPUP_LIVE

    if POPUP and POPUP.winfo_exists():
        POPUP.destroy()
        POPUP = None
    if POPUP_LIVE and POPUP_LIVE.winfo_exists():
        POPUP_LIVE.destroy()
        POPUP_LIVE = None


def update_pop_status(text: str) -> None:
    if popup_status_label is not None:
        popup_status_label.configure(text=tr(text))


def update_pop_live_status(text: str) -> None:
    if popup_status_label_live is not None:
        popup_status_label_live.configure(text=tr(text))


def update_tumbler(var: str, value: bool) -> None:
    modules.globals.fp_ui[var] = value
    save_switch_states()
    # If we're currently in a live preview, update the frame processors
    if PREVIEW.state() == "normal":
        global frame_processors
        frame_processors = get_frame_processors_modules(
            modules.globals.frame_processors
        )


def select_source_path() -> None:
    global RECENT_DIRECTORY_SOURCE, img_ft, vid_ft

    PREVIEW.withdraw()
    source_path = ctk.filedialog.askopenfilename(
        title=tr("select an source image"),
        initialdir=RECENT_DIRECTORY_SOURCE,
        filetypes=[img_ft],
    )
    if is_image(source_path):
        modules.globals.source_path = source_path
        RECENT_DIRECTORY_SOURCE = os.path.dirname(modules.globals.source_path)
        image = render_image_preview(modules.globals.source_path, (200, 200))
        source_label.configure(image=image)
    else:
        modules.globals.source_path = None
        source_label.configure(image=None)


def swap_faces_paths() -> None:
    global RECENT_DIRECTORY_SOURCE, RECENT_DIRECTORY_TARGET

    source_path = modules.globals.source_path
    target_path = modules.globals.target_path

    if not is_image(source_path) or not is_image(target_path):
        return

    modules.globals.source_path = target_path
    modules.globals.target_path = source_path

    RECENT_DIRECTORY_SOURCE = os.path.dirname(modules.globals.source_path)
    RECENT_DIRECTORY_TARGET = os.path.dirname(modules.globals.target_path)

    PREVIEW.withdraw()

    source_image = render_image_preview(modules.globals.source_path, (200, 200))
    source_label.configure(image=source_image)

    target_image = render_image_preview(modules.globals.target_path, (200, 200))
    target_label.configure(image=target_image)


def select_target_path() -> None:
    global RECENT_DIRECTORY_TARGET, img_ft, vid_ft

    PREVIEW.withdraw()
    target_path = ctk.filedialog.askopenfilename(
        title=tr("select an target image or video"),
        initialdir=RECENT_DIRECTORY_TARGET,
        filetypes=[img_ft, vid_ft],
    )
    if is_image(target_path):
        modules.globals.target_path = target_path
        RECENT_DIRECTORY_TARGET = os.path.dirname(modules.globals.target_path)
        image = render_image_preview(modules.globals.target_path, (200, 200))
        target_label.configure(image=image)
    elif is_video(target_path):
        modules.globals.target_path = target_path
        RECENT_DIRECTORY_TARGET = os.path.dirname(modules.globals.target_path)
        video_frame = render_video_preview(target_path, (200, 200))
        target_label.configure(image=video_frame)
    else:
        modules.globals.target_path = None
        target_label.configure(image=None)


def select_output_path(start: Callable[[], None]) -> None:
    global RECENT_DIRECTORY_OUTPUT, img_ft, vid_ft

    if is_image(modules.globals.target_path):
        output_path = ctk.filedialog.asksaveasfilename(
            title=tr("save image output file"),
            filetypes=[img_ft],
            defaultextension=".png",
            initialfile="output.png",
            initialdir=RECENT_DIRECTORY_OUTPUT,
        )
    elif is_video(modules.globals.target_path):
        output_path = ctk.filedialog.asksaveasfilename(
            title=tr("save video output file"),
            filetypes=[vid_ft],
            defaultextension=".mp4",
            initialfile="output.mp4",
            initialdir=RECENT_DIRECTORY_OUTPUT,
        )
    else:
        output_path = None
    if output_path:
        modules.globals.output_path = output_path
        RECENT_DIRECTORY_OUTPUT = os.path.dirname(modules.globals.output_path)
        start()


def check_and_ignore_nsfw(target, destroy: Callable = None) -> bool:
    """Check if the target is NSFW.
    TODO: Consider to make blur the target.
    """
    from numpy import ndarray
    from modules.predicter import predict_image, predict_video, predict_frame

    if type(target) is str:  # image/video file path
        check_nsfw = predict_image if has_image_extension(target) else predict_video
    elif type(target) is ndarray:  # frame object
        check_nsfw = predict_frame
    if check_nsfw and check_nsfw(target):
        if destroy:
            destroy(
                to_quit=False
            )  # Do not need to destroy the window frame if the target is NSFW
        update_status("Processing ignored!")
        return True
    else:
        return False


def fit_image_to_size(image, width: int, height: int):
    if image is None:
        return image

    h, w, _ = image.shape

    if not width or width <= 0:
        width = w
    if not height or height <= 0:
        height = h

    width_ratio = width / w
    height_ratio = height / h
    ratio = min(width_ratio, height_ratio)

    if ratio <= 0:
        return image

    new_width = max(1, int(round(w * ratio)))
    new_height = max(1, int(round(h * ratio)))

    if new_width == w and new_height == h:
        return image

    interpolation = cv2.INTER_CUBIC if ratio > 1 else cv2.INTER_AREA
    return cv2.resize(image, (new_width, new_height), interpolation=interpolation)


def render_image_preview(image_path: str, size: Tuple[int, int]) -> ctk.CTkImage:
    image = Image.open(image_path)
    if size:
        image = ImageOps.fit(image, size, Image.LANCZOS)
    return ctk.CTkImage(image, size=image.size)


def render_video_preview(
    video_path: str, size: Tuple[int, int], frame_number: int = 0
) -> ctk.CTkImage:
    capture = cv2.VideoCapture(video_path)
    if frame_number:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    has_frame, frame = capture.read()
    if has_frame:
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if size:
            image = ImageOps.fit(image, size, Image.LANCZOS)
        return ctk.CTkImage(image, size=image.size)
    capture.release()
    cv2.destroyAllWindows()


def toggle_preview() -> None:
    if PREVIEW.state() == "normal":
        PREVIEW.withdraw()
    elif modules.globals.source_path and modules.globals.target_path:
        init_preview()
        update_preview()


def init_preview() -> None:
    if is_image(modules.globals.target_path):
        preview_slider.pack_forget()
    if is_video(modules.globals.target_path):
        video_frame_total = get_video_frame_total(modules.globals.target_path)
        preview_slider.configure(to=video_frame_total)
        preview_slider.pack(fill="x")
        preview_slider.set(0)


def update_preview(frame_number: int = 0) -> None:
    if modules.globals.source_path and modules.globals.target_path:
        update_status("Processing...")
        temp_frame = get_video_frame(modules.globals.target_path, frame_number)
        if modules.globals.nsfw_filter and check_and_ignore_nsfw(temp_frame):
            return
        for frame_processor in get_frame_processors_modules(
            modules.globals.frame_processors
        ):
            temp_frame = frame_processor.process_frame(
                get_one_face(read_static_image(modules.globals.source_path)), temp_frame
            )
        image = Image.fromarray(cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB))
        image = ImageOps.contain(
            image, (PREVIEW_MAX_WIDTH, PREVIEW_MAX_HEIGHT), Image.LANCZOS
        )
        image = ctk.CTkImage(image, size=image.size)
        preview_label.configure(image=image)
        update_status("Processing succeed!")
        PREVIEW.deiconify()


def webcam_preview(root: ctk.CTk, camera_index: int):
    global POPUP_LIVE

    if POPUP_LIVE and POPUP_LIVE.winfo_exists():
        update_status("Source x Target Mapper is already open.")
        POPUP_LIVE.focus()
        return

    if not modules.globals.map_faces:
        if modules.globals.source_path is None:
            update_status("Please select a source image first")
            return
        create_webcam_preview(camera_index)
    else:
        modules.globals.souce_target_map = []
        create_source_target_popup_for_webcam(
            root, modules.globals.souce_target_map, camera_index
        )


def get_available_cameras():
    """Returns a list of available camera names and indices."""
    camera_indices = []
    camera_names = []

    for camera in enumerate_cameras():
        cap = cv2.VideoCapture(camera.index)
        if cap.isOpened():
            camera_indices.append(camera.index)
            camera_names.append(camera.name)
            cap.release()
    return (camera_indices, camera_names)


def _capture_thread_func(
    camera: VideoCapturer, capture_queue: "queue.Queue[Any]", stop_event: threading.Event
) -> None:
    while not stop_event.is_set():
        ret, frame = camera.read()
        if not ret:
            stop_event.set()
            break
        try:
            capture_queue.put_nowait(frame)
        except queue.Full:
            try:
                capture_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                capture_queue.put_nowait(frame)
            except queue.Full:
                pass


def _processing_thread_func(
    capture_queue: "queue.Queue[Any]",
    processed_queue: "queue.Queue[Any]",
    stop_event: threading.Event,
) -> None:
    frame_processors = get_frame_processors_modules(modules.globals.frame_processors)
    source_image = None
    source_path_cache: Optional[str] = None
    prev_time = time.time()
    fps_update_interval = 0.5
    frame_count = 0
    fps = 0.0
    proc_frame_index = 0
    cached_target_face = None
    cached_many_faces: Optional[List[Any]] = None

    while not stop_event.is_set():
        try:
            frame = capture_queue.get(timeout=0.05)
        except queue.Empty:
            continue

        temp_frame = frame.copy()
        run_detection = proc_frame_index % DETECT_EVERY_N == 0
        proc_frame_index += 1

        if modules.globals.live_mirror:
            temp_frame = cv2.flip(temp_frame, 1)

        if not modules.globals.map_faces:
            current_source_path = modules.globals.source_path
            if current_source_path != source_path_cache:
                source_path_cache = current_source_path
                if current_source_path:
                    source_image = get_one_face(read_static_image(current_source_path))
                else:
                    source_image = None

            if run_detection or (
                cached_target_face is None and cached_many_faces is None
            ):
                if modules.globals.many_faces:
                    detected_faces = get_many_faces(temp_frame)
                    cached_many_faces = (
                        detected_faces if detected_faces is not None else []
                    )
                    cached_target_face = None
                else:
                    cached_target_face = get_one_face(temp_frame)
                    cached_many_faces = None

            for frame_processor in frame_processors:
                try:
                    if frame_processor.NAME == "DLC.FACE-ENHANCER":
                        if modules.globals.fp_ui["face_enhancer"]:
                            temp_frame = frame_processor.process_frame(None, temp_frame)
                    elif frame_processor.NAME == "DLC.FACE-SWAPPER":
                        swapped_face_bboxes = []
                        if modules.globals.many_faces and cached_many_faces:
                            result = temp_frame.copy()
                            for target_face in cached_many_faces:
                                result = frame_processor.swap_face(
                                    source_image, target_face, result
                                )
                                bbox = getattr(target_face, "bbox", None)
                                if bbox is not None:
                                    swapped_face_bboxes.append(
                                        [
                                            int(bbox[0]),
                                            int(bbox[1]),
                                            int(bbox[2]),
                                            int(bbox[3]),
                                        ]
                                    )
                            temp_frame = result
                        elif cached_target_face is not None:
                            temp_frame = frame_processor.swap_face(
                                source_image, cached_target_face, temp_frame
                            )
                            bbox = getattr(cached_target_face, "bbox", None)
                            if bbox is not None:
                                swapped_face_bboxes.append(
                                    [
                                        int(bbox[0]),
                                        int(bbox[1]),
                                        int(bbox[2]),
                                        int(bbox[3]),
                                    ]
                                )

                        if hasattr(frame_processor, "apply_post_processing"):
                            temp_frame = frame_processor.apply_post_processing(
                                temp_frame, swapped_face_bboxes
                            )
                    else:
                        temp_frame = frame_processor.process_frame(
                            source_image, temp_frame
                        )
                except Exception:
                    continue
        else:
            modules.globals.target_path = None
            for frame_processor in frame_processors:
                try:
                    if frame_processor.NAME == "DLC.FACE-ENHANCER":
                        if modules.globals.fp_ui["face_enhancer"]:
                            temp_frame = frame_processor.process_frame_v2(temp_frame)
                    else:
                        temp_frame = frame_processor.process_frame_v2(temp_frame)
                except Exception:
                    continue

        current_time = time.time()
        frame_count += 1
        if current_time - prev_time >= fps_update_interval:
            fps = frame_count / (current_time - prev_time)
            frame_count = 0
            prev_time = current_time

        if modules.globals.show_fps:
            cv2.putText(
                temp_frame,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

        if modules.globals.virtual_camera:
            modules.virtual_camera.send(temp_frame)

        try:
            processed_queue.put_nowait(temp_frame)
        except queue.Full:
            try:
                processed_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                processed_queue.put_nowait(temp_frame)
            except queue.Full:
                pass


def create_webcam_preview(camera_index: int):
    global preview_label, PREVIEW

    try:
        camera = VideoCapturer(camera_index)
    except Exception:
        update_status("Failed to start camera")
        return

    if not camera.start(modules.globals.live_width, modules.globals.live_height, 60):
        update_status("Failed to start camera")
        return

    PREVIEW.deiconify()

    if modules.globals.virtual_camera:
        modules.virtual_camera.init(
            modules.globals.live_width, modules.globals.live_height, 60
        )

    capture_queue: "queue.Queue[Any]" = queue.Queue(maxsize=2)
    processed_queue: "queue.Queue[Any]" = queue.Queue(maxsize=2)
    stop_event = threading.Event()

    capture_thread = threading.Thread(
        target=_capture_thread_func,
        args=(camera, capture_queue, stop_event),
        daemon=True,
    )
    processing_thread = threading.Thread(
        target=_processing_thread_func,
        args=(capture_queue, processed_queue, stop_event),
        daemon=True,
    )

    capture_thread.start()
    processing_thread.start()

    try:
        while not stop_event.is_set():
            try:
                temp_frame = processed_queue.get(timeout=0.03)
            except queue.Empty:
                ROOT.update()
                if PREVIEW.state() == "withdrawn":
                    break
                continue

            PREVIEW.update_idletasks()
            available_width = max(1, PREVIEW.winfo_width())
            available_height = max(1, PREVIEW.winfo_height())

            if modules.globals.live_resizable:
                temp_frame = fit_image_to_size(
                    temp_frame, available_width, available_height
                )

            image = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image = ImageOps.contain(
                image, (temp_frame.shape[1], temp_frame.shape[0]), Image.LANCZOS
            )
            image = ctk.CTkImage(image, size=image.size)
            preview_label.configure(image=image)
            ROOT.update()

            if PREVIEW.state() == "withdrawn":
                break
    finally:
        stop_event.set()
        capture_thread.join(timeout=2.0)
        processing_thread.join(timeout=2.0)
        camera.release()
        if modules.globals.virtual_camera:
            modules.virtual_camera.stop()
        PREVIEW.withdraw()


def create_source_target_popup_for_webcam(
    root: ctk.CTk, map: list, camera_index: int
) -> None:
    global POPUP_LIVE, popup_status_label_live

    POPUP_LIVE = ctk.CTkToplevel(root)
    POPUP_LIVE.title(tr("Source x Target Mapper"))
    POPUP_LIVE.geometry(f"{POPUP_LIVE_WIDTH}x{POPUP_LIVE_HEIGHT}")
    POPUP_LIVE.focus()

    def on_submit_click():
        if has_valid_map():
            POPUP_LIVE.destroy()
            simplify_maps()
            create_webcam_preview(camera_index)
        else:
            update_pop_live_status("At least 1 source with target is required!")

    def on_add_click():
        add_blank_map()
        refresh_data(map)
        update_pop_live_status("Please provide mapping!")

    def on_clear_click():
        clear_source_target_images(map)
        refresh_data(map)
        update_pop_live_status("All mappings cleared!")

    popup_status_label_live = ctk.CTkLabel(
        POPUP_LIVE,
        text=None,
        justify="center",
        font=ctk.CTkFont(size=14, weight="normal")
    )
    popup_status_label_live.grid(row=1, column=0, pady=16)

    add_button = ctk.CTkButton(
        POPUP_LIVE,
        text=tr("Add"),
        command=lambda: on_add_click(),
        font=ctk.CTkFont(size=14, weight="normal"),
        fg_color="#4A4D50",
        hover_color="#6E7174"
    )
    add_button.place(relx=0.1, rely=0.92, relwidth=0.2, relheight=0.05)

    clear_button = ctk.CTkButton(
        POPUP_LIVE,
        text=tr("Clear"),
        command=lambda: on_clear_click(),
        font=ctk.CTkFont(size=14, weight="normal"),
        fg_color="#4A4D50",
        hover_color="#6E7174"
    )
    clear_button.place(relx=0.4, rely=0.92, relwidth=0.2, relheight=0.05)

    close_button = ctk.CTkButton(
        POPUP_LIVE,
        text=tr("Submit"),
        command=lambda: on_submit_click(),
        font=ctk.CTkFont(size=15, weight="bold"),
        fg_color="#6366F1",
        hover_color="#8B5CF6"
    )
    close_button.place(relx=0.7, rely=0.92, relwidth=0.2, relheight=0.05)


def clear_source_target_images(map: list) -> None:
    global source_label_dict_live, target_label_dict_live

    for item in map:
        if "source" in item:
            del item["source"]
        if "target" in item:
            del item["target"]

    for button_num in list(source_label_dict_live.keys()):
        source_label_dict_live[button_num].destroy()
        del source_label_dict_live[button_num]
    for button_num in list(target_label_dict_live.keys()):
        target_label_dict_live[button_num].destroy()
        del target_label_dict_live[button_num]


def refresh_data(map: list):
    global POPUP_LIVE

    scrollable_frame = ctk.CTkScrollableFrame(
        POPUP_LIVE, width=POPUP_LIVE_SCROLL_WIDTH, height=POPUP_LIVE_SCROLL_HEIGHT
    )
    scrollable_frame.grid(row=0, column=0, padx=16, pady=16, sticky="nsew")

    def on_sbutton_click(map, button_num):
        map = update_webcam_source(scrollable_frame, map, button_num)

    def on_tbutton_click(map, button_num):
        map = update_webcam_target(scrollable_frame, map, button_num)

    for item in map:
        id = item["id"]

        button = ctk.CTkButton(
            scrollable_frame,
            text=tr("Select source image"),
            command=lambda id=id: on_sbutton_click(map, id),
            width=DEFAULT_BUTTON_WIDTH,
            height=DEFAULT_BUTTON_HEIGHT,
            font=ctk.CTkFont(size=14, weight="normal")
        )
        button.grid(row=id, column=0, padx=30, pady=12)

        x_label = ctk.CTkLabel(
            scrollable_frame,
            text=f"X",
            width=MAPPER_PREVIEW_MAX_WIDTH,
            height=MAPPER_PREVIEW_MAX_HEIGHT,
        )
        x_label.grid(row=id, column=2, padx=10, pady=10)

        button = ctk.CTkButton(
            scrollable_frame,
            text=tr("Select target image"),
            command=lambda id=id: on_tbutton_click(map, id),
            width=DEFAULT_BUTTON_WIDTH,
            height=DEFAULT_BUTTON_HEIGHT,
            font=ctk.CTkFont(size=14, weight="normal")
        )
        button.grid(row=id, column=3, padx=20, pady=12)

        if "source" in item:
            image = Image.fromarray(
                cv2.cvtColor(item["source"]["cv2"], cv2.COLOR_BGR2RGB)
            )
            image = image.resize(
                (MAPPER_PREVIEW_MAX_WIDTH, MAPPER_PREVIEW_MAX_HEIGHT), Image.LANCZOS
            )
            tk_image = ctk.CTkImage(image, size=image.size)

            source_image = ctk.CTkLabel(
                scrollable_frame,
                text=f"S-{id}",
                width=MAPPER_PREVIEW_MAX_WIDTH,
                height=MAPPER_PREVIEW_MAX_HEIGHT,
            )
            source_image.grid(row=id, column=1, padx=10, pady=10)
            source_image.configure(image=tk_image)

        if "target" in item:
            image = Image.fromarray(
                cv2.cvtColor(item["target"]["cv2"], cv2.COLOR_BGR2RGB)
            )
            image = image.resize(
                (MAPPER_PREVIEW_MAX_WIDTH, MAPPER_PREVIEW_MAX_HEIGHT), Image.LANCZOS
            )
            tk_image = ctk.CTkImage(image, size=image.size)

            target_image = ctk.CTkLabel(
                scrollable_frame,
                text=f"T-{id}",
                width=MAPPER_PREVIEW_MAX_WIDTH,
                height=MAPPER_PREVIEW_MAX_HEIGHT,
            )
            target_image.grid(row=id, column=4, padx=20, pady=10)
            target_image.configure(image=tk_image)


def update_webcam_source(
    scrollable_frame: ctk.CTkScrollableFrame, map: list, button_num: int
) -> list:
    global source_label_dict_live

    source_path = ctk.filedialog.askopenfilename(
        title=tr("select an source image"),
        initialdir=RECENT_DIRECTORY_SOURCE,
        filetypes=[img_ft],
    )

    if "source" in map[button_num]:
        map[button_num].pop("source")
        source_label_dict_live[button_num].destroy()
        del source_label_dict_live[button_num]

    if source_path == "":
        return map
    else:
        cv2_img = read_static_image(source_path)
        face = get_one_face(cv2_img)

        if face:
            x_min, y_min, x_max, y_max = face["bbox"]

            map[button_num]["source"] = {
                "cv2": cv2_img[int(y_min) : int(y_max), int(x_min) : int(x_max)],
                "face": face,
            }

            image = Image.fromarray(
                cv2.cvtColor(map[button_num]["source"]["cv2"], cv2.COLOR_BGR2RGB)
            )
            image = image.resize(
                (MAPPER_PREVIEW_MAX_WIDTH, MAPPER_PREVIEW_MAX_HEIGHT), Image.LANCZOS
            )
            tk_image = ctk.CTkImage(image, size=image.size)

            source_image = ctk.CTkLabel(
                scrollable_frame,
                text=f"S-{button_num}",
                width=MAPPER_PREVIEW_MAX_WIDTH,
                height=MAPPER_PREVIEW_MAX_HEIGHT,
            )
            source_image.grid(row=button_num, column=1, padx=10, pady=10)
            source_image.configure(image=tk_image)
            source_label_dict_live[button_num] = source_image
        else:
            update_pop_live_status("Face could not be detected in last upload!")
        return map


def update_webcam_target(
    scrollable_frame: ctk.CTkScrollableFrame, map: list, button_num: int
) -> list:
    global target_label_dict_live

    target_path = ctk.filedialog.askopenfilename(
        title=tr("select an target image"),
        initialdir=RECENT_DIRECTORY_SOURCE,
        filetypes=[img_ft],
    )

    if "target" in map[button_num]:
        map[button_num].pop("target")
        target_label_dict_live[button_num].destroy()
        del target_label_dict_live[button_num]

    if target_path == "":
        return map
    else:
        cv2_img = read_static_image(target_path)
        face = get_one_face(cv2_img)

        if face:
            x_min, y_min, x_max, y_max = face["bbox"]

            map[button_num]["target"] = {
                "cv2": cv2_img[int(y_min) : int(y_max), int(x_min) : int(x_max)],
                "face": face,
            }

            image = Image.fromarray(
                cv2.cvtColor(map[button_num]["target"]["cv2"], cv2.COLOR_BGR2RGB)
            )
            image = image.resize(
                (MAPPER_PREVIEW_MAX_WIDTH, MAPPER_PREVIEW_MAX_HEIGHT), Image.LANCZOS
            )
            tk_image = ctk.CTkImage(image, size=image.size)

            target_image = ctk.CTkLabel(
                scrollable_frame,
                text=f"T-{button_num}",
                width=MAPPER_PREVIEW_MAX_WIDTH,
                height=MAPPER_PREVIEW_MAX_HEIGHT,
            )
            target_image.grid(row=button_num, column=4, padx=20, pady=10)
            target_image.configure(image=tk_image)
            target_label_dict_live[button_num] = target_image
        else:
            update_pop_live_status("Face could not be detected in last upload!")
        return map
