import pyvirtualcam
import numpy as np
import cv2

_virtual_cam = None
_target_width = 0
_target_height = 0

def init(width, height, fps):
    global _virtual_cam, _target_width, _target_height
    try:
        _virtual_cam = pyvirtualcam.Camera(width=width, height=height, fps=fps, fmt=pyvirtualcam.PixelFormat.BGR)
        _target_width = width
        _target_height = height
        print(f'Virtual camera started: {_virtual_cam.device}')
        return True
    except Exception as e:
        print(f'Failed to start virtual camera: {e}')
        return False

def send(frame):
    global _virtual_cam, _target_width, _target_height
    if _virtual_cam:
        try:
            if frame.shape[1] != _target_width or frame.shape[0] != _target_height:
                frame = cv2.resize(frame, (_target_width, _target_height))
            _virtual_cam.send(frame)
        except Exception as e:
            print(f"Virtual camera send error: {e}")

def stop():
    global _virtual_cam
    if _virtual_cam:
        _virtual_cam.close()
        _virtual_cam = None
