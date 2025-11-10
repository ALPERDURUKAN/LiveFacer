from typing import Any, List
import cv2
import insightface
import threading
import numpy as np
import modules.globals
import modules.processors.frame.core
from modules.core import update_status
from modules.face_analyser import get_one_face, get_many_faces, default_source_face
from modules.typing import Face, Frame
from modules.utilities import (
    conditional_download,
    resolve_relative_path,
    is_image,
    is_video,
)
from modules.cluster_analysis import find_closest_centroid

FACE_SWAPPER = None
THREAD_LOCK = threading.Lock()
NAME = "DLC.FACE-SWAPPER"
PREVIOUS_FRAME_RESULT = None


def pre_check() -> bool:
    download_directory_path = resolve_relative_path("../models")
    conditional_download(
        download_directory_path,
        [
            "https://huggingface.co/hacksider/deep-live-cam/blob/main/inswapper_128_fp16.onnx"
        ],
    )
    return True


def pre_start() -> bool:
    if not modules.globals.map_faces and not is_image(modules.globals.source_path):
        update_status("Select an image for source path.", NAME)
        return False
    elif not modules.globals.map_faces and not get_one_face(
        cv2.imread(modules.globals.source_path)
    ):
        update_status("No face in source path detected.", NAME)
        return False
    if not is_image(modules.globals.target_path) and not is_video(
        modules.globals.target_path
    ):
        update_status("Select an image or video for target path.", NAME)
        return False
    return True


def get_face_swapper() -> Any:
    global FACE_SWAPPER

    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            model_path = resolve_relative_path("../models/inswapper_128_fp16.onnx")
            FACE_SWAPPER = insightface.model_zoo.get_model(
                model_path, providers=modules.globals.execution_providers
            )
    return FACE_SWAPPER


def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    face_swapper = get_face_swapper()

    original_frame = temp_frame.copy()

    # Apply the face swap
    swapped_frame_raw = face_swapper.get(
        temp_frame, target_face, source_face, paste_back=True
    )
    
    swapped_frame_raw = np.clip(swapped_frame_raw, 0, 255).astype(np.uint8)
    swapped_frame_raw = cv2.resize(swapped_frame_raw, (temp_frame.shape[1], temp_frame.shape[0]))
    swapped_frame = np.clip(swapped_frame_raw, 0, 255).astype(np.uint8)

    if modules.globals.mouth_mask:
        # Create a mask for the target face
        face_mask = create_face_mask(target_face, temp_frame)

        # Create the mouth mask
        mouth_mask, mouth_cutout, mouth_box, lower_lip_polygon = (
            create_lower_mouth_mask(target_face, temp_frame)
        )

        # Apply the mouth area
        swapped_frame = apply_mouth_area(
            swapped_frame, mouth_cutout, mouth_box, face_mask, lower_lip_polygon
        )

        if modules.globals.show_mouth_mask_box:
            mouth_mask_data = (mouth_mask, mouth_cutout, mouth_box, lower_lip_polygon)
            swapped_frame = draw_mouth_mask_visualization(
                swapped_frame, target_face, mouth_mask_data
            )

    # Apply opacity blending
    opacity = getattr(modules.globals, "opacity", 1.0)
    opacity = max(0.0, min(1.0, opacity))
    final_swapped_frame = cv2.addWeighted(
        original_frame.astype(np.uint8), 1 - opacity, 
        swapped_frame.astype(np.uint8), opacity, 0
    )
    final_swapped_frame = final_swapped_frame.astype(np.uint8)

    return final_swapped_frame


def apply_post_processing(current_frame: Frame, swapped_face_bboxes: List[np.ndarray]) -> Frame:
    global PREVIOUS_FRAME_RESULT
    
    processed_frame = current_frame.copy()
    
    # 1. Apply Sharpness (if enabled)
    sharpness_value = getattr(modules.globals, "sharpness", 0.0)
    if sharpness_value > 0.0 and swapped_face_bboxes:
        try:
            for bbox in swapped_face_bboxes:
                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                # Ensure valid coordinates
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(processed_frame.shape[1], int(x2)), min(processed_frame.shape[0], int(y2))
                
                if x2 > x1 and y2 > y1:
                    face_region = processed_frame[y1:y2, x1:x2]
                    blurred = cv2.GaussianBlur(face_region, (0, 0), 3)
                    sharpened_region = cv2.addWeighted(
                        face_region, 1.0 + sharpness_value, blurred, -sharpness_value, 0
                    )
                    sharpened_region = np.clip(sharpened_region, 0, 255).astype(np.uint8)
                    processed_frame[y1:y2, x1:x2] = sharpened_region
        except Exception:
            # Skip sharpening for this region if it fails
            pass
    
    # 2. Apply Interpolation (if enabled)
    enable_interpolation = getattr(modules.globals, "enable_interpolation", False)
    interpolation_weight = getattr(modules.globals, "interpolation_weight", 0.2)
    final_frame = processed_frame  # Start with the current (potentially sharpened) frame
    
    if enable_interpolation and PREVIOUS_FRAME_RESULT is not None:
        try:
            # Check if previous frame matches current frame dimensions
            if (PREVIOUS_FRAME_RESULT.shape == processed_frame.shape and 
                PREVIOUS_FRAME_RESULT.dtype == processed_frame.dtype):
                # Perform interpolation
                final_frame = cv2.addWeighted(
                    PREVIOUS_FRAME_RESULT.astype(np.float32), interpolation_weight,
                    processed_frame.astype(np.float32), 1.0 - interpolation_weight, 0
                )
                # Ensure final frame is uint8
                final_frame = np.clip(final_frame, 0, 255).astype(np.uint8)
        except Exception:
            final_frame = processed_frame  # Use current frame if interpolation fails
            PREVIOUS_FRAME_RESULT = None  # Reset state if error occurs
    
    # Update the state for the next frame *with the interpolated result*
    PREVIOUS_FRAME_RESULT = final_frame.copy()
    
    return final_frame


def process_frame(source_face: Face, temp_frame: Frame) -> Frame:
    if modules.globals.color_correction:
        temp_frame = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB)

    processed_frame = temp_frame
    swapped_face_bboxes = []
    
    if modules.globals.many_faces:
        many_faces = get_many_faces(temp_frame)
        if many_faces:
            current_swap_target = processed_frame.copy()
            for target_face in many_faces:
                current_swap_target = swap_face(source_face, target_face, current_swap_target)
                # Collect bounding boxes for post-processing
                bbox = target_face.bbox
                swapped_face_bboxes.append([int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])])
            processed_frame = current_swap_target
    else:
        target_face = get_one_face(temp_frame)
        if target_face:
            processed_frame = swap_face(source_face, target_face, processed_frame)
            # Collect bounding box for post-processing
            bbox = target_face.bbox
            swapped_face_bboxes.append([int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])])
    
    final_frame = apply_post_processing(processed_frame, swapped_face_bboxes)
    
    return final_frame


def process_frame_v2(temp_frame: Frame, temp_frame_path: str = "") -> Frame:
    processed_frame = temp_frame
    swapped_face_bboxes = []
    
    if is_image(modules.globals.target_path):
        if modules.globals.many_faces:
            source_face = default_source_face()
            for map in modules.globals.souce_target_map:
                target_face = map["target"]["face"]
                processed_frame = swap_face(source_face, target_face, processed_frame)
                bbox = target_face.bbox
                swapped_face_bboxes.append([int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])])

        elif not modules.globals.many_faces:
            for map in modules.globals.souce_target_map:
                if "source" in map:
                    source_face = map["source"]["face"]
                    target_face = map["target"]["face"]
                    processed_frame = swap_face(source_face, target_face, processed_frame)
                    bbox = target_face.bbox
                    swapped_face_bboxes.append([int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])])

    elif is_video(modules.globals.target_path):
        if modules.globals.many_faces:
            source_face = default_source_face()
            for map in modules.globals.souce_target_map:
                target_frame = [
                    f
                    for f in map["target_faces_in_frame"]
                    if f["location"] == temp_frame_path
                ]

                for frame in target_frame:
                    for target_face in frame["faces"]:
                        processed_frame = swap_face(source_face, target_face, processed_frame)
                        bbox = target_face.bbox
                        swapped_face_bboxes.append([int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])])

        elif not modules.globals.many_faces:
            for map in modules.globals.souce_target_map:
                if "source" in map:
                    target_frame = [
                        f
                        for f in map["target_faces_in_frame"]
                        if f["location"] == temp_frame_path
                    ]
                    source_face = map["source"]["face"]

                    for frame in target_frame:
                        for target_face in frame["faces"]:
                            processed_frame = swap_face(source_face, target_face, processed_frame)
                            bbox = target_face.bbox
                            swapped_face_bboxes.append([int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])])

    else:
        detected_faces = get_many_faces(processed_frame)
        if modules.globals.many_faces:
            if detected_faces:
                source_face = default_source_face()
                for target_face in detected_faces:
                    processed_frame = swap_face(source_face, target_face, processed_frame)
                    bbox = target_face.bbox
                    swapped_face_bboxes.append([int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])])

        elif not modules.globals.many_faces:
            if detected_faces:
                if len(detected_faces) <= len(
                    modules.globals.simple_map["target_embeddings"]
                ):
                    for detected_face in detected_faces:
                        closest_centroid_index, _ = find_closest_centroid(
                            modules.globals.simple_map["target_embeddings"],
                            detected_face.normed_embedding,
                        )

                        processed_frame = swap_face(
                            modules.globals.simple_map["source_faces"][
                                closest_centroid_index
                            ],
                            detected_face,
                            processed_frame,
                        )
                        bbox = detected_face.bbox
                        swapped_face_bboxes.append([int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])])
                else:
                    detected_faces_centroids = []
                    for face in detected_faces:
                        detected_faces_centroids.append(face.normed_embedding)
                    i = 0
                    for target_embedding in modules.globals.simple_map[
                        "target_embeddings"
                    ]:
                        closest_centroid_index, _ = find_closest_centroid(
                            detected_faces_centroids, target_embedding
                        )

                        processed_frame = swap_face(
                            modules.globals.simple_map["source_faces"][i],
                            detected_faces[closest_centroid_index],
                            processed_frame,
                        )
                        bbox = detected_faces[closest_centroid_index].bbox
                        swapped_face_bboxes.append([int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])])
                        i += 1
    
    final_frame = apply_post_processing(processed_frame, swapped_face_bboxes)
    return final_frame


def process_frames(
    source_path: str, temp_frame_paths: List[str], progress: Any = None
) -> None:
    if not modules.globals.map_faces:
        source_face = get_one_face(cv2.imread(source_path))
        for temp_frame_path in temp_frame_paths:
            temp_frame = cv2.imread(temp_frame_path)
            try:
                result = process_frame(source_face, temp_frame)
                cv2.imwrite(temp_frame_path, result)
            except Exception as exception:
                print(exception)
                pass
            if progress:
                progress.update(1)
    else:
        for temp_frame_path in temp_frame_paths:
            temp_frame = cv2.imread(temp_frame_path)
            try:
                result = process_frame_v2(temp_frame, temp_frame_path)
                cv2.imwrite(temp_frame_path, result)
            except Exception as exception:
                print(exception)
                pass
            if progress:
                progress.update(1)


def process_image(source_path: str, target_path: str, output_path: str) -> None:
    if not modules.globals.map_faces:
        source_face = get_one_face(cv2.imread(source_path))
        target_frame = cv2.imread(target_path)
        result = process_frame(source_face, target_frame)
        cv2.imwrite(output_path, result)
    else:
        if modules.globals.many_faces:
            update_status(
                "Many faces enabled. Using first source image. Progressing...", NAME
            )
        target_frame = cv2.imread(output_path)
        result = process_frame_v2(target_frame)
        cv2.imwrite(output_path, result)


def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    global PREVIOUS_FRAME_RESULT
    # Reset interpolation state before starting video processing
    PREVIOUS_FRAME_RESULT = None
    
    if modules.globals.map_faces and modules.globals.many_faces:
        update_status(
            "Many faces enabled. Using first source image. Progressing...", NAME
        )
    modules.processors.frame.core.process_video(
        source_path, temp_frame_paths, process_frames
    )


def create_lower_mouth_mask(
    face: Face, frame: Frame
) -> (np.ndarray, np.ndarray, tuple, np.ndarray):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    mouth_cutout = None
    lower_lip_polygon = None  # Initialize
    mouth_box = (0, 0, 0, 0)  # Initialize
    
    # Validate face and landmarks
    if face is None:
        return mask, mouth_cutout, mouth_box, lower_lip_polygon
    
    landmarks = face.landmark_2d_106
    # Check landmark validity
    if landmarks is None or landmarks.shape[0] < 106:
        return mask, mouth_cutout, mouth_box, lower_lip_polygon
    
    try:  # Wrap main logic in try-except
        #                  0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20
        lower_lip_order = [65, 66, 62, 70, 69, 18, 19, 20, 21, 22, 23, 24, 0, 8, 7, 6, 5, 4, 3, 2, 65]  # 21 points
        # Check if all indices are valid for the loaded landmarks
        if max(lower_lip_order) >= landmarks.shape[0]:
            return mask, mouth_cutout, mouth_box, lower_lip_polygon
        
        lower_lip_landmarks = landmarks[lower_lip_order].astype(np.float32)
        # Filter out potential NaN or Inf values in landmarks
        if not np.all(np.isfinite(lower_lip_landmarks)):
            return mask, mouth_cutout, mouth_box, lower_lip_polygon
        
        center = np.mean(lower_lip_landmarks, axis=0)
        if not np.all(np.isfinite(center)):  # Check center calculation
            return mask, mouth_cutout, mouth_box, lower_lip_polygon
        
        mask_down_size = getattr(modules.globals, "mask_down_size", 0.1)  # Default 0.1
        expansion_factor = 1 + mask_down_size
        expanded_landmarks = (lower_lip_landmarks - center) * expansion_factor + center
        mask_size = getattr(modules.globals, "mask_size", 1.0)  # Default 1.0
        toplip_extension = mask_size * 0.5
        # Define toplip indices relative to lower_lip_order (safer)
        toplip_local_indices = [0, 1, 2, 3, 4, 5, 19]  # Indices in lower_lip_order for [65, 66, 62, 70, 69, 18, 2]
        for idx in toplip_local_indices:
            if idx < len(expanded_landmarks):  # Boundary check
                direction = expanded_landmarks[idx] - center
                norm = np.linalg.norm(direction)
                if norm > 1e-6:  # Avoid division by zero
                    direction_normalized = direction / norm
                    expanded_landmarks[idx] += direction_normalized * toplip_extension
        
        # Define chin indices relative to lower_lip_order
        chin_local_indices = [9, 10, 11, 12, 13, 14]  # Indices for [22, 23, 24, 0, 8, 7]
        chin_extension = 2 * 0.2
        for idx in chin_local_indices:
            if idx < len(expanded_landmarks):
                # Extend vertically based on distance from center y
                y_diff = expanded_landmarks[idx][1] - center[1]
                expanded_landmarks[idx][1] += y_diff * chin_extension
        
        # Ensure landmarks are finite after adjustments
        if not np.all(np.isfinite(expanded_landmarks)):
            return mask, mouth_cutout, mouth_box, lower_lip_polygon
        
        expanded_landmarks = expanded_landmarks.astype(np.int32)
        
        # Calculate bounding box
        min_x, min_y = np.min(expanded_landmarks, axis=0)
        max_x, max_y = np.max(expanded_landmarks, axis=0)
        
        # Add padding *after* initial min/max calculation
        padding_ratio = 0.1  # Percentage padding
        padding_x = int((max_x - min_x) * padding_ratio)
        padding_y = int((max_y - min_y) * padding_ratio)  # Use y-range for y-padding
        # Apply padding and clamp to frame boundaries
        frame_w, frame_h = frame.shape[1], frame.shape[0]
        min_x = max(0, min_x - padding_x)
        min_y = max(0, min_y - padding_y)
        max_x = min(frame_w, max_x + padding_x)
        max_y = min(frame_h, max_y + padding_y)
        
        # Create the mask ROI
        mask_roi_h = max_y - min_y
        mask_roi_w = max_x - min_x
        mask_roi = np.zeros((mask_roi_h, mask_roi_w), dtype=np.uint8)
        # Shift polygon coordinates relative to the ROI's top-left corner
        polygon_relative_to_roi = expanded_landmarks - [min_x, min_y]
        cv2.fillPoly(mask_roi, [polygon_relative_to_roi], 255)
        
        blur_k_size = getattr(modules.globals, "mask_blur_kernel", 15)
        blur_k_size = max(1, blur_k_size // 2 * 2 + 1)  # Ensure odd
        mask_roi = cv2.GaussianBlur(mask_roi, (blur_k_size, blur_k_size), 0)
        
        # Place the mask ROI in the full-sized mask
        mask[min_y:max_y, min_x:max_x] = mask_roi
        
        mouth_cutout = frame[min_y:max_y, min_x:max_x].copy()
        lower_lip_polygon = expanded_landmarks
        mouth_box = (min_x, min_y, max_x, max_y)
    except Exception as e:
        # Return defaults on error
        pass
    
    return mask, mouth_cutout, mouth_box, lower_lip_polygon


def draw_mouth_mask_visualization(
    frame: Frame, face: Face, mouth_mask_data: tuple
) -> Frame:
    landmarks = face.landmark_2d_106
    if landmarks is not None and mouth_mask_data is not None:
        mask, mouth_cutout, (min_x, min_y, max_x, max_y), lower_lip_polygon = (
            mouth_mask_data
        )

        vis_frame = frame.copy()

        # Ensure coordinates are within frame bounds
        height, width = vis_frame.shape[:2]
        min_x, min_y = max(0, min_x), max(0, min_y)
        max_x, max_y = min(width, max_x), min(height, max_y)

        # Adjust mask to match the region size
        mask_region = mask[0 : max_y - min_y, 0 : max_x - min_x]

        # Remove the color mask overlay
        # color_mask = cv2.applyColorMap((mask_region * 255).astype(np.uint8), cv2.COLORMAP_JET)

        # Ensure shapes match before blending
        vis_region = vis_frame[min_y:max_y, min_x:max_x]
        # Remove blending with color_mask
        # if vis_region.shape[:2] == color_mask.shape[:2]:
        #     blended = cv2.addWeighted(vis_region, 0.7, color_mask, 0.3, 0)
        #     vis_frame[min_y:max_y, min_x:max_x] = blended

        # Draw the lower lip polygon
        cv2.polylines(vis_frame, [lower_lip_polygon], True, (0, 255, 0), 2)

        # Remove the red box
        # cv2.rectangle(vis_frame, (min_x, min_y), (max_x, max_y), (0, 0, 255), 2)

        # Visualize the feathered mask
        feather_amount = max(
            1,
            min(
                30,
                (max_x - min_x) // modules.globals.mask_feather_ratio,
                (max_y - min_y) // modules.globals.mask_feather_ratio,
            ),
        )
        # Ensure kernel size is odd
        kernel_size = 2 * feather_amount + 1
        feathered_mask = cv2.GaussianBlur(
            mask_region.astype(float), (kernel_size, kernel_size), 0
        )
        feathered_mask = (feathered_mask / feathered_mask.max() * 255).astype(np.uint8)
        # Remove the feathered mask color overlay
        # color_feathered_mask = cv2.applyColorMap(feathered_mask, cv2.COLORMAP_VIRIDIS)

        # Ensure shapes match before blending feathered mask
        # if vis_region.shape == color_feathered_mask.shape:
        #     blended_feathered = cv2.addWeighted(vis_region, 0.7, color_feathered_mask, 0.3, 0)
        #     vis_frame[min_y:max_y, min_x:max_x] = blended_feathered

        # Add labels
        cv2.putText(
            vis_frame,
            "Lower Mouth Mask",
            (min_x, min_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            vis_frame,
            "Feathered Mask",
            (min_x, max_y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        return vis_frame
    return frame


def apply_mouth_area(
    frame: np.ndarray,
    mouth_cutout: np.ndarray,
    mouth_box: tuple,
    face_mask: np.ndarray,
    mouth_polygon: np.ndarray,
) -> np.ndarray:
    if frame is None or len(frame.shape) != 3:
        return frame
    
    min_x, min_y, max_x, max_y = mouth_box
    box_width = max_x - min_x
    box_height = max_y - min_y

    if (
        mouth_cutout is None
        or box_width <= 0
        or box_height <= 0
        or face_mask is None
        or mouth_polygon is None
        or len(mouth_polygon) == 0
    ):
        return frame

    try:
        # Validate bounds
        frame_h, frame_w = frame.shape[:2]
        min_x = max(0, int(min_x))
        min_y = max(0, int(min_y))
        max_x = min(frame_w, int(max_x))
        max_y = min(frame_h, int(max_y))
        
        if max_x <= min_x or max_y <= min_y:
            return frame
        
        roi = frame[min_y:max_y, min_x:max_x]
        if roi.size == 0:
            return frame
        
        resized_mouth_cutout = None
        try:
            resized_mouth_cutout = cv2.resize(mouth_cutout, (box_width, box_height), interpolation=cv2.INTER_LINEAR)
        except Exception:
            resized_mouth_cutout = mouth_cutout
        
        if resized_mouth_cutout is None or resized_mouth_cutout.size == 0:
            return frame
        
        if roi.shape != resized_mouth_cutout.shape:
            try:
                resized_mouth_cutout = cv2.resize(
                    resized_mouth_cutout, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_LINEAR
                )
            except Exception:
                return frame

        color_corrected_mouth = resized_mouth_cutout
        try:
            color_corrected_mouth = apply_color_transfer(resized_mouth_cutout, roi)
        except Exception:
            color_corrected_mouth = resized_mouth_cutout

        # Use the provided mouth polygon to create the mask
        polygon_mask_roi = np.zeros(roi.shape[:2], dtype=np.uint8)
        adjusted_polygon = mouth_polygon - [min_x, min_y]
        # Draw the filled polygon on the ROI mask
        cv2.fillPoly(polygon_mask_roi, [adjusted_polygon.astype(np.int32)], 255)
        
        # Feather the polygon mask (Gaussian blur)
        mask_feather_ratio = getattr(modules.globals, "mask_feather_ratio", 12)  # Default 12
        # Calculate feather amount based on the smaller dimension of the box
        feather_base_dim = min(box_width, box_height)
        feather_amount = max(1, min(30, feather_base_dim // max(1, mask_feather_ratio)))  # Avoid div by zero
        # Ensure kernel size is odd and positive
        kernel_size = 2 * feather_amount + 1
        feathered_polygon_mask = cv2.GaussianBlur(polygon_mask_roi.astype(float), (kernel_size, kernel_size), 0)
        # Normalize feathered mask to [0.0, 1.0] range
        max_val = feathered_polygon_mask.max()
        if max_val > 1e-6:  # Avoid division by zero
            feathered_polygon_mask = feathered_polygon_mask / max_val
        else:
            feathered_polygon_mask.fill(0.0)  # Mask is all black if max is near zero
        
        # --- End Mask Creation ---
        # --- Refined Blending ---
        # Get the corresponding ROI from the *full face mask* (already blurred)
        # Ensure face_mask is float and normalized [0.0, 1.0]
        if face_mask.dtype == np.uint8:
            face_mask_float = face_mask.astype(float) / 255.0
        else:  # Assume already float [0,1] if type is float
            face_mask_float = face_mask
        face_mask_roi = face_mask_float[min_y:max_y, min_x:max_x]
        # Combine the feathered mouth polygon mask with the face mask ROI
        # Use minimum to ensure we only affect area inside both masks (mouth area within face)
        # This helps blend the edges smoothly with the surrounding swapped face region
        combined_mask = np.minimum(feathered_polygon_mask, face_mask_roi)
        # Expand mask to 3 channels for blending (ensure it matches image channels)
        if len(roi.shape) == 3:
            combined_mask_3channel = combined_mask[:, :, np.newaxis]
            # Ensure data types are compatible for blending (float or double for mask, uint8 for images)
            color_corrected_mouth_uint8 = color_corrected_mouth.astype(np.uint8)
            roi_uint8 = roi.astype(np.uint8)
            combined_mask_float = combined_mask_3channel.astype(np.float64)  # Use float64 for precision in mask
            # Blend: (original_mouth * combined_mask) + (swapped_face_roi * (1 - combined_mask))
            blended_roi = (color_corrected_mouth_uint8 * combined_mask_float +
                          roi_uint8 * (1 - combined_mask_float)).astype(np.uint8)
            # Place the blended ROI back into the frame
            frame[min_y:max_y, min_x:max_x] = blended_roi
        else:
            # Don't modify frame if it's not BGR
            pass
    except Exception as e:
        # Don't crash, just return the frame as is
        pass

    return frame


def create_face_mask(face: Face, frame: Frame) -> np.ndarray:
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    landmarks = face.landmark_2d_106
    if landmarks is not None:
        landmarks_int = landmarks.astype(np.int32)
        face_outline_points = landmarks_int[0:33]
        
        # Calculate padding
        padding = int(
            np.linalg.norm(face_outline_points[0] - face_outline_points[16]) * 0.05
        )  # 5% of face width
        
        # Create a slightly larger convex hull for padding
        full_face_poly = face_outline_points
        hull = cv2.convexHull(full_face_poly.astype(np.float32))
        
        # Fill the convex hull
        cv2.fillConvexPoly(mask, hull.astype(np.int32), 255)
        
        # Smooth the mask edges
        blur_k_size = getattr(modules.globals, "face_mask_blur", 31)
        blur_k_size = max(1, blur_k_size // 2 * 2 + 1)  # Ensure odd
        mask = cv2.GaussianBlur(mask, (blur_k_size, blur_k_size), 0)

    return mask


def apply_color_transfer(source, target):
    """
    Apply color transfer from target to source image
    """
    try:
        # Handle grayscale images
        if len(source.shape) == 2:
            source = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
            source = np.clip(source, 0, 255).astype(np.uint8)
        if len(target.shape) == 2:
            target = cv2.cvtColor(target, cv2.COLOR_GRAY2BGR)
            target = np.clip(target, 0, 255).astype(np.uint8)
        
        # Ensure both are BGR
        if source.shape[2] != 3 or target.shape[2] != 3:
            return source
        
        result_bgr = source
        
        try:
            source_float = source.astype(np.float32) / 255.0
            target_float = target.astype(np.float32) / 255.0
            source_lab = cv2.cvtColor(source_float, cv2.COLOR_BGR2LAB)
            target_lab = cv2.cvtColor(target_float, cv2.COLOR_BGR2LAB)
            
            source_mean, source_std = cv2.meanStdDev(source_lab)
            target_mean, target_std = cv2.meanStdDev(target_lab)
            
            source_mean = source_mean.reshape((1, 1, 3))
            source_std = source_std.reshape((1, 1, 3))
            target_mean = target_mean.reshape((1, 1, 3))
            target_std = target_std.reshape((1, 1, 3))
            epsilon = 1e-6
            source_std = np.maximum(source_std, epsilon)
            result_lab = (source_lab - source_mean) * (target_std / source_std) + target_mean
            result_bgr_float = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)
            result_bgr_float = np.clip(result_bgr_float, 0.0, 1.0)
            result_bgr = (result_bgr_float * 255.0).astype("uint8")
        except Exception:
            result_bgr = source
        
        return result_bgr
    except Exception:
        return source
