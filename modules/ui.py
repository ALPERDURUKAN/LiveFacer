import os
import webbrowser
import customtkinter as ctk
from typing import Callable, Tuple
import cv2
from cv2_enumerate_cameras import enumerate_cameras  # Add this import
from PIL import Image, ImageOps
import time
import json

import modules.globals
import modules.metadata
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
from modules.utilities import (
    is_image,
    is_video,
    resolve_relative_path,
    has_image_extension,
)

ROOT = None
POPUP = None
POPUP_LIVE = None
ROOT_HEIGHT = 850
ROOT_WIDTH = 680

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

DEFAULT_BUTTON_WIDTH = 200
DEFAULT_BUTTON_HEIGHT = 40

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
        "color_correction": modules.globals.color_correction,
        "nsfw_filter": modules.globals.nsfw_filter,
        "live_mirror": modules.globals.live_mirror,
        "live_resizable": modules.globals.live_resizable,
        "fp_ui": modules.globals.fp_ui,
        "show_fps": modules.globals.show_fps,
        "mouth_mask": modules.globals.mouth_mask,
        "show_mouth_mask_box": modules.globals.show_mouth_mask_box,
        "eyes_mask": modules.globals.eyes_mask,
        "show_eyes_mask_box": modules.globals.show_eyes_mask_box,
        "eyebrows_mask": modules.globals.eyebrows_mask,
        "show_eyebrows_mask_box": modules.globals.show_eyebrows_mask_box,
        "live_width": modules.globals.live_width,
        "live_height": modules.globals.live_height,
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
        modules.globals.color_correction = switch_states.get("color_correction", False)
        modules.globals.nsfw_filter = switch_states.get("nsfw_filter", False)
        modules.globals.live_mirror = switch_states.get("live_mirror", False)
        modules.globals.live_resizable = switch_states.get("live_resizable", False)
        modules.globals.fp_ui = switch_states.get("fp_ui", {"face_enhancer": False})
        modules.globals.show_fps = switch_states.get("show_fps", False)
        modules.globals.mouth_mask = switch_states.get("mouth_mask", False)
        modules.globals.show_mouth_mask_box = switch_states.get("show_mouth_mask_box", False)
        modules.globals.eyes_mask = switch_states.get("eyes_mask", False)
        modules.globals.show_eyes_mask_box = switch_states.get("show_eyes_mask_box", False)
        modules.globals.eyebrows_mask = switch_states.get("eyebrows_mask", False)
        modules.globals.show_eyebrows_mask_box = switch_states.get("show_eyebrows_mask_box", False)
        modules.globals.live_width = switch_states.get("live_width", 640)
        modules.globals.live_height = switch_states.get("live_height", 640)
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
    content.grid(row=0, column=0, sticky="nsew", padx=32, pady=(32, 16))
    content.grid_columnconfigure(0, weight=1)
    content.grid_rowconfigure(2, weight=1)

    face_preview_frame = ctk.CTkFrame(content, fg_color="transparent")
    face_preview_frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))
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
        width=220,
        height=220,
        corner_radius=12,
        fg_color="transparent"
    )
    source_label.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

    swap_icon = ctk.CTkLabel(
        face_preview_frame,
        text="↔",
        font=ctk.CTkFont(size=36, weight="bold"),
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
        width=220,
        height=220,
        corner_radius=12,
        fg_color="transparent"
    )
    target_label.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

    button_frame = ctk.CTkFrame(content, fg_color="transparent")
    button_frame.grid(row=1, column=0, sticky="ew", pady=(16, 24))
    button_frame.grid_columnconfigure((0, 1, 2), weight=1, uniform="face-buttons")

    select_face_button = ctk.CTkButton(
        button_frame,
        text="Select a face",
        cursor="hand2",
        command=select_source_path,
        font=ctk.CTkFont(size=15, weight="normal")
    )
    select_face_button.grid(row=0, column=0, sticky="ew", padx=(0, 12))

    swap_faces_button = ctk.CTkButton(
        button_frame,
        text="↔",
        cursor="hand2",
        command=swap_faces_paths,
        width=56,
        font=ctk.CTkFont(size=20, weight="bold")
    )
    swap_faces_button.grid(row=0, column=1, sticky="ew", padx=12)

    select_target_button = ctk.CTkButton(
        button_frame,
        text="Select a target",
        cursor="hand2",
        command=select_target_path,
        font=ctk.CTkFont(size=15, weight="normal")
    )
    select_target_button.grid(row=0, column=2, sticky="ew", padx=(12, 0))

    switch_frame = ctk.CTkFrame(content, corner_radius=16)
    switch_frame.grid(row=2, column=0, sticky="ew", pady=(0, 24))
    switch_frame.grid_columnconfigure((0, 1), weight=1, uniform="switch-columns")

    eyebrows_mask_var = ctk.BooleanVar(value=modules.globals.eyebrows_mask)
    eyebrows_mask_switch = ctk.CTkSwitch(
        switch_frame,
        text="Eyebrows Mask",
        variable=eyebrows_mask_var,
        cursor="hand2",
        command=lambda: (
            setattr(modules.globals, "eyebrows_mask", eyebrows_mask_var.get()),
            save_switch_states(),
        ),
    )
    eyebrows_mask_switch.grid(row=0, column=0, sticky="w", padx=(24, 16), pady=8)

    eyes_mask_var = ctk.BooleanVar(value=modules.globals.eyes_mask)
    eyes_mask_switch = ctk.CTkSwitch(
        switch_frame,
        text="Eyes Mask",
        variable=eyes_mask_var,
        cursor="hand2",
        command=lambda: (
            setattr(modules.globals, "eyes_mask", eyes_mask_var.get()),
            save_switch_states(),
        ),
    )
    eyes_mask_switch.grid(row=1, column=0, sticky="w", padx=(24, 16), pady=8)

    mouth_mask_var = ctk.BooleanVar(value=modules.globals.mouth_mask)
    mouth_mask_switch = ctk.CTkSwitch(
        switch_frame,
        text="Mouth Mask",
        variable=mouth_mask_var,
        cursor="hand2",
        command=lambda: (
            setattr(modules.globals, "mouth_mask", mouth_mask_var.get()),
            save_switch_states(),
        ),
    )
    mouth_mask_switch.grid(row=2, column=0, sticky="w", padx=(24, 16), pady=8)

    keep_fps_value = ctk.BooleanVar(value=modules.globals.keep_fps)
    keep_fps_switch = ctk.CTkSwitch(
        switch_frame,
        text="Keep FPS",
        variable=keep_fps_value,
        cursor="hand2",
        command=lambda: (
            setattr(modules.globals, "keep_fps", keep_fps_value.get()),
            save_switch_states(),
        ),
    )
    keep_fps_switch.grid(row=3, column=0, sticky="w", padx=(24, 16), pady=8)

    keep_frames_value = ctk.BooleanVar(value=modules.globals.keep_frames)
    keep_frames_switch = ctk.CTkSwitch(
        switch_frame,
        text="Keep Frames",
        variable=keep_frames_value,
        cursor="hand2",
        command=lambda: (
            setattr(modules.globals, "keep_frames", keep_frames_value.get()),
            save_switch_states(),
        ),
    )
    keep_frames_switch.grid(row=4, column=0, sticky="w", padx=(24, 16), pady=8)

    enhancer_value = ctk.BooleanVar(value=modules.globals.fp_ui["face_enhancer"])
    enhancer_switch = ctk.CTkSwitch(
        switch_frame,
        text="Face Enhancer",
        variable=enhancer_value,
        cursor="hand2",
        command=lambda: (
            update_tumbler("face_enhancer", enhancer_value.get()),
            save_switch_states(),
        ),
    )
    enhancer_switch.grid(row=5, column=0, sticky="w", padx=(24, 16), pady=8)

    map_faces_var = ctk.BooleanVar(value=modules.globals.map_faces)
    map_faces_switch = ctk.CTkSwitch(
        switch_frame,
        text="Map Faces",
        variable=map_faces_var,
        cursor="hand2",
        command=lambda: (
            setattr(modules.globals, "map_faces", map_faces_var.get()),
            save_switch_states(),
        ),
    )
    map_faces_switch.grid(row=6, column=0, sticky="w", padx=(24, 16), pady=8)

    show_eyebrows_mask_box_var = ctk.BooleanVar(
        value=modules.globals.show_eyebrows_mask_box
    )
    show_eyebrows_mask_box_switch = ctk.CTkSwitch(
        switch_frame,
        text="Show Eyebrows Mask Box",
        variable=show_eyebrows_mask_box_var,
        cursor="hand2",
        command=lambda: (
            setattr(
                modules.globals,
                "show_eyebrows_mask_box",
                show_eyebrows_mask_box_var.get(),
            ),
            save_switch_states(),
        ),
    )
    show_eyebrows_mask_box_switch.grid(row=0, column=1, sticky="w", padx=(16, 24), pady=8)

    show_eyes_mask_box_var = ctk.BooleanVar(value=modules.globals.show_eyes_mask_box)
    show_eyes_mask_box_switch = ctk.CTkSwitch(
        switch_frame,
        text="Show Eyes Mask Box",
        variable=show_eyes_mask_box_var,
        cursor="hand2",
        command=lambda: (
            setattr(
                modules.globals, "show_eyes_mask_box", show_eyes_mask_box_var.get()
            ),
            save_switch_states(),
        ),
    )
    show_eyes_mask_box_switch.grid(row=1, column=1, sticky="w", padx=(16, 24), pady=8)

    show_mouth_mask_box_var = ctk.BooleanVar(
        value=modules.globals.show_mouth_mask_box
    )
    show_mouth_mask_box_switch = ctk.CTkSwitch(
        switch_frame,
        text="Show Mouth Mask Box",
        variable=show_mouth_mask_box_var,
        cursor="hand2",
        command=lambda: (
            setattr(
                modules.globals,
                "show_mouth_mask_box",
                show_mouth_mask_box_var.get(),
            ),
            save_switch_states(),
        ),
    )
    show_mouth_mask_box_switch.grid(row=2, column=1, sticky="w", padx=(16, 24), pady=8)

    keep_audio_value = ctk.BooleanVar(value=modules.globals.keep_audio)
    keep_audio_switch = ctk.CTkSwitch(
        switch_frame,
        text="Keep Audio",
        variable=keep_audio_value,
        cursor="hand2",
        command=lambda: (
            setattr(modules.globals, "keep_audio", keep_audio_value.get()),
            save_switch_states(),
        ),
    )
    keep_audio_switch.grid(row=3, column=1, sticky="w", padx=(16, 24), pady=8)

    many_faces_value = ctk.BooleanVar(value=modules.globals.many_faces)
    many_faces_switch = ctk.CTkSwitch(
        switch_frame,
        text="Many Faces",
        variable=many_faces_value,
        cursor="hand2",
        command=lambda: (
            setattr(modules.globals, "many_faces", many_faces_value.get()),
            save_switch_states(),
        ),
    )
    many_faces_switch.grid(row=4, column=1, sticky="w", padx=(16, 24), pady=8)

    color_correction_value = ctk.BooleanVar(value=modules.globals.color_correction)
    color_correction_switch = ctk.CTkSwitch(
        switch_frame,
        text="Fix Blueish Cam",
        variable=color_correction_value,
        cursor="hand2",
        command=lambda: (
            setattr(
                modules.globals, "color_correction", color_correction_value.get()
            ),
            save_switch_states(),
        ),
    )
    color_correction_switch.grid(row=5, column=1, sticky="w", padx=(16, 24), pady=8)

    show_fps_value = ctk.BooleanVar(value=modules.globals.show_fps)
    show_fps_switch = ctk.CTkSwitch(
        switch_frame,
        text="Show FPS",
        variable=show_fps_value,
        cursor="hand2",
        command=lambda: (
            setattr(modules.globals, "show_fps", show_fps_value.get()),
            save_switch_states(),
        ),
    )
    show_fps_switch.grid(row=6, column=1, sticky="w", padx=(16, 24), pady=8)

    slider_frame = ctk.CTkFrame(content, fg_color="transparent")
    slider_frame.grid(row=3, column=0, sticky="ew", pady=(0, 24))
    slider_frame.grid_columnconfigure((0, 1), weight=1, uniform="sliders")

    sharpness_container = ctk.CTkFrame(slider_frame, fg_color="transparent")
    sharpness_container.grid(row=0, column=0, sticky="ew", padx=(0, 16))
    sharpness_container.grid_columnconfigure(0, weight=1)

    sharpness_var = ctk.DoubleVar(value=modules.globals.sharpness)

    def on_sharpness_change(value: float):
        val = float(value)
        modules.globals.sharpness = val
        percentage = int(val * 100)
        sharpness_label.configure(text=f"Sharpness: {percentage}%")

    sharpness_label = ctk.CTkLabel(
        sharpness_container,
        text=f"Sharpness: {int(modules.globals.sharpness * 100)}%",
        font=ctk.CTkFont(size=15, weight="normal")
    )
    sharpness_label.grid(row=0, column=0, sticky="w", pady=(0, 4))

    sharpness_slider = ctk.CTkSlider(
        sharpness_container,
        from_=0.0,
        to=1.0,
        variable=sharpness_var,
        command=on_sharpness_change,
    )
    sharpness_slider.grid(row=1, column=0, sticky="ew", pady=(8, 0))

    opacity_container = ctk.CTkFrame(slider_frame, fg_color="transparent")
    opacity_container.grid(row=0, column=1, sticky="ew", padx=(16, 0))
    opacity_container.grid_columnconfigure(0, weight=1)

    opacity_var = ctk.DoubleVar(value=modules.globals.opacity)

    def on_opacity_change(value: float):
        val = float(value)
        modules.globals.opacity = val
        percentage = int(val * 100)
        opacity_label.configure(text=f"Opacity: {percentage}%")

    opacity_label = ctk.CTkLabel(
        opacity_container,
        text=f"Opacity: {int(modules.globals.opacity * 100)}%",
        font=ctk.CTkFont(size=15, weight="normal")
    )
    opacity_label.grid(row=0, column=0, sticky="w", pady=(0, 4))

    opacity_slider = ctk.CTkSlider(
        opacity_container,
        from_=0.0,
        to=1.0,
        variable=opacity_var,
        command=on_opacity_change,
    )
    opacity_slider.grid(row=1, column=0, sticky="ew", pady=(8, 0))

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
            text=f"Enhancer Intensity: {percentage}%"
        )

    enhancer_intensity_label = ctk.CTkLabel(
        enhancer_intensity_container,
        text=(
            f"Enhancer Intensity: {int(modules.globals.face_enhancer_intensity * 100)}%"
        ),
        font=ctk.CTkFont(size=15, weight="normal"),
    )
    enhancer_intensity_label.grid(row=0, column=0, sticky="w", pady=(0, 4))

    enhancer_intensity_slider = ctk.CTkSlider(
        enhancer_intensity_container,
        from_=0.0,
        to=1.0,
        variable=enhancer_intensity_var,
        command=on_enhancer_intensity_change,
    )
    enhancer_intensity_slider.grid(row=1, column=0, sticky="ew", pady=(8, 0))

    camera_frame = ctk.CTkFrame(content, fg_color="transparent")
    camera_frame.grid(row=4, column=0, sticky="ew", pady=(0, 24))
    camera_frame.grid_columnconfigure(1, weight=1)
    camera_frame.grid_columnconfigure(3, weight=1)

    camera_label = ctk.CTkLabel(
        camera_frame,
        text="Select Camera:",
        font=ctk.CTkFont(size=15, weight="normal")
    )
    camera_label.grid(row=0, column=0, sticky="w", padx=(0, 16))

    available_camera_indices, available_camera_strings = get_available_cameras()
    camera_values = (
        available_camera_strings if available_camera_strings else ["No cameras found"]
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
        text="Live",
        cursor="hand2",
        state="normal" if available_camera_strings else "disabled",
        command=on_live_click,
    )
    live_button.grid(row=0, column=2, padx=(16, 0))

    # Live Resolution Controls
    resolution_label = ctk.CTkLabel(
        camera_frame,
        text="Live Resolution:",
        font=ctk.CTkFont(size=15, weight="normal")
    )
    resolution_label.grid(row=1, column=0, sticky="w", padx=(0, 16), pady=(16, 0))

    def validate_resolution(value):
        try:
            val = int(value)
            return 320 <= val <= 3840
        except:
            return False

    def on_width_change(value):
        if validate_resolution(value):
            modules.globals.live_width = int(value)
            save_switch_states()

    def on_height_change(value):
        if validate_resolution(value):
            modules.globals.live_height = int(value)
            save_switch_states()

    width_label = ctk.CTkLabel(
        camera_frame,
        text="Width:",
        font=ctk.CTkFont(size=14, weight="normal")
    )
    width_label.grid(row=1, column=1, sticky="w", padx=(0, 8), pady=(16, 0))

    width_entry = ctk.CTkEntry(
        camera_frame,
        width=80,
        placeholder_text=str(modules.globals.live_width)
    )
    width_entry.insert(0, str(modules.globals.live_width))
    width_entry.grid(row=1, column=2, padx=(0, 16), pady=(16, 0), sticky="w")
    width_entry.bind("<Return>", lambda e: on_width_change(width_entry.get()))
    width_entry.bind("<FocusOut>", lambda e: on_width_change(width_entry.get()))

    height_label = ctk.CTkLabel(
        camera_frame,
        text="Height:",
        font=ctk.CTkFont(size=14, weight="normal")
    )
    height_label.grid(row=1, column=3, sticky="w", padx=(0, 8), pady=(16, 0))

    height_entry = ctk.CTkEntry(
        camera_frame,
        width=80,
        placeholder_text=str(modules.globals.live_height)
    )
    height_entry.insert(0, str(modules.globals.live_height))
    height_entry.grid(row=1, column=4, padx=(0, 0), pady=(16, 0), sticky="w")
    height_entry.bind("<Return>", lambda e: on_height_change(height_entry.get()))
    height_entry.bind("<FocusOut>", lambda e: on_height_change(height_entry.get()))

    action_frame = ctk.CTkFrame(content, fg_color="transparent")
    action_frame.grid(row=5, column=0, sticky="ew", pady=(24, 0))
    action_frame.grid_columnconfigure((0, 1, 2), weight=1, uniform="action-buttons")

    start_button = ctk.CTkButton(
        action_frame,
        text="Start",
        cursor="hand2",
        command=lambda: analyze_target(start, root),
        font=ctk.CTkFont(size=16, weight="bold"),
        fg_color="#6366F1",
        hover_color="#8B5CF6"
    )
    start_button.grid(row=0, column=0, sticky="ew", padx=(0, 12))

    stop_button = ctk.CTkButton(
        action_frame,
        text="Destroy",
        cursor="hand2",
        command=destroy,
        font=ctk.CTkFont(size=15, weight="normal"),
        fg_color="#4A4D50",
        hover_color="#6E7174"
    )
    stop_button.grid(row=0, column=1, sticky="ew", padx=12)

    preview_button = ctk.CTkButton(
        action_frame,
        text="Preview",
        cursor="hand2",
        command=toggle_preview,
        font=ctk.CTkFont(size=15, weight="normal"),
        fg_color="#6366F1",
        hover_color="#8B5CF6"
    )
    preview_button.grid(row=0, column=2, sticky="ew", padx=(12, 0))

    donate_label = ctk.CTkLabel(
        content,
        text="LiveFacer",
        justify="center",
        cursor="hand2",
        font=ctk.CTkFont(size=16, weight="normal")
    )
    donate_label.grid(row=6, column=0, sticky="ew", pady=(16, 0))
    donate_label.configure(
        text_color=ctk.ThemeManager.theme.get("URL").get("text_color")
    )
    donate_label.bind(
        "<Button>", lambda event: webbrowser.open(modules.metadata.website)
    )

    status_label = ctk.CTkLabel(
        content,
        text=None,
        justify="center",
        font=ctk.CTkFont(size=14, weight="normal")
    )
    status_label.grid(row=7, column=0, sticky="ew", pady=(8, 0))

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
    POPUP.title("Source x Target Mapper")
    POPUP.geometry(f"{POPUP_WIDTH}x{POPUP_HEIGHT}")
    POPUP.focus()

    def on_submit_click(start):
        if has_valid_map():
            POPUP.destroy()
            select_output_path(start)
        else:
            update_pop_status("Atleast 1 source with target is required!")

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
            text="Select source image",
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
        text="Submit",
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
        title="select an source image",
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
        cv2_img = cv2.imread(source_path)
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
    preview.title("Preview")
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
    status_label.configure(text=text)
    ROOT.update()


def update_pop_status(text: str) -> None:
    popup_status_label.configure(text=text)


def update_pop_live_status(text: str) -> None:
    popup_status_label_live.configure(text=text)


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
        title="select an source image",
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
        title="select an target image or video",
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
            title="save image output file",
            filetypes=[img_ft],
            defaultextension=".png",
            initialfile="output.png",
            initialdir=RECENT_DIRECTORY_OUTPUT,
        )
    elif is_video(modules.globals.target_path):
        output_path = ctk.filedialog.asksaveasfilename(
            title="save video output file",
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
    if width is None and height is None:
        return image
    h, w, _ = image.shape
    ratio_h = 0.0
    ratio_w = 0.0
    if width > height:
        ratio_h = height / h
    else:
        ratio_w = width / w
    ratio = max(ratio_w, ratio_h)
    new_size = (int(ratio * w), int(ratio * h))
    return cv2.resize(image, dsize=new_size)


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
                get_one_face(cv2.imread(modules.globals.source_path)), temp_frame
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
    if not modules.globals.map_faces:
        if modules.globals.source_path is None:
            # No image selected
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


def create_webcam_preview(camera_index: int):
    global preview_label, PREVIEW

    camera = cv2.VideoCapture(camera_index)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, modules.globals.live_width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, modules.globals.live_height)
    camera.set(cv2.CAP_PROP_FPS, 60)

    preview_label.configure(width=modules.globals.live_width, height=modules.globals.live_height)

    PREVIEW.deiconify()

    frame_processors = get_frame_processors_modules(modules.globals.frame_processors)

    source_image = None
    prev_time = time.time()
    fps_update_interval = 0.5  # Update FPS every 0.5 seconds
    frame_count = 0
    fps = 0

    while camera:
        ret, frame = camera.read()
        if not ret:
            break

        temp_frame = frame.copy()

        if modules.globals.live_mirror:
            temp_frame = cv2.flip(temp_frame, 1)

        if modules.globals.live_resizable:
            temp_frame = fit_image_to_size(
                temp_frame, PREVIEW.winfo_width(), PREVIEW.winfo_height()
            )

        if not modules.globals.map_faces:
            if source_image is None and modules.globals.source_path:
                source_image = get_one_face(cv2.imread(modules.globals.source_path))

            for frame_processor in frame_processors:
                if frame_processor.NAME == "DLC.FACE-ENHANCER":
                    if modules.globals.fp_ui["face_enhancer"]:
                        temp_frame = frame_processor.process_frame(None, temp_frame)
                else:
                    temp_frame = frame_processor.process_frame(source_image, temp_frame)
        else:
            modules.globals.target_path = None

            for frame_processor in frame_processors:
                if frame_processor.NAME == "DLC.FACE-ENHANCER":
                    if modules.globals.fp_ui["face_enhancer"]:
                        temp_frame = frame_processor.process_frame_v2(temp_frame)
                else:
                    temp_frame = frame_processor.process_frame_v2(temp_frame)

        # Calculate and display FPS
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

    camera.release()
    PREVIEW.withdraw()


def create_source_target_popup_for_webcam(
    root: ctk.CTk, map: list, camera_index: int
) -> None:
    global POPUP_LIVE, popup_status_label_live

    POPUP_LIVE = ctk.CTkToplevel(root)
    POPUP_LIVE.title("Source x Target Mapper")
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

    popup_status_label_live = ctk.CTkLabel(
        POPUP_LIVE,
        text=None,
        justify="center",
        font=ctk.CTkFont(size=14, weight="normal")
    )
    popup_status_label_live.grid(row=1, column=0, pady=16)

    add_button = ctk.CTkButton(
        POPUP_LIVE,
        text="Add",
        command=lambda: on_add_click(),
        font=ctk.CTkFont(size=14, weight="normal"),
        fg_color="#4A4D50",
        hover_color="#6E7174"
    )
    add_button.place(relx=0.2, rely=0.92, relwidth=0.2, relheight=0.05)

    close_button = ctk.CTkButton(
        POPUP_LIVE,
        text="Submit",
        command=lambda: on_submit_click(),
        font=ctk.CTkFont(size=15, weight="bold"),
        fg_color="#6366F1",
        hover_color="#8B5CF6"
    )
    close_button.place(relx=0.6, rely=0.92, relwidth=0.2, relheight=0.05)


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
            text="Select source image",
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
            text="Select target image",
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
        title="select an source image",
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
        cv2_img = cv2.imread(source_path)
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
        title="select an target image",
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
        cv2_img = cv2.imread(target_path)
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
