import cv2
import numpy as np
from typing import Optional, Tuple, Callable
import platform

if platform.system() == "Windows":
    try:
        from pygrabber.dshow_graph import FilterGraph
    except ImportError:
        FilterGraph = None


class VideoCapturer:
    def __init__(self, device_index: int):
        self.device_index = device_index
        self.frame_callback: Optional[Callable[[np.ndarray], None]] = None
        self.is_running = False
        self.cap = None

        if platform.system() == "Windows" and FilterGraph is not None:
            self.graph = FilterGraph()
            devices = self.graph.get_input_devices()
            if self.device_index < 0 or self.device_index >= len(devices):
                raise ValueError(
                    f"Invalid device index {device_index}. Available devices: {len(devices)}"
                )

    def start(self, width: int = 960, height: int = 540, fps: int = 60) -> bool:
        try:
            if platform.system() == "Windows":
                capture_methods = [
                    (self.device_index, cv2.CAP_DSHOW),
                    (self.device_index, cv2.CAP_ANY),
                    (-1, cv2.CAP_ANY),
                    (0, cv2.CAP_ANY),
                ]
                for dev_id, backend in capture_methods:
                    try:
                        self.cap = cv2.VideoCapture(dev_id, backend)
                        if self.cap.isOpened():
                            break
                        self.cap.release()
                    except Exception:
                        continue
            else:
                self.cap = cv2.VideoCapture(self.device_index)

            if not self.cap or not self.cap.isOpened():
                raise RuntimeError("Failed to open camera")

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, fps)
            self.is_running = True
            return True
        except Exception as exception:
            print(f"Failed to start capture: {exception}")
            if self.cap:
                self.cap.release()
            return False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self.is_running or self.cap is None:
            return False, None

        ret, frame = self.cap.read()
        if ret:
            if self.frame_callback:
                self.frame_callback(frame)
            return True, frame
        return False, None

    def release(self) -> None:
        if self.is_running and self.cap is not None:
            self.cap.release()
            self.is_running = False
            self.cap = None

    def set_frame_callback(self, callback: Callable[[np.ndarray], None]) -> None:
        self.frame_callback = callback
