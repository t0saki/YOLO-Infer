"""
Visualization utilities for YOLO11 project.

This module provides functions for drawing detections, creating videos,
and other visualization tasks.
"""

import cv2
import numpy as np
import torch
from typing import Optional, List, Tuple, Dict, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def draw_detections(
    image: np.ndarray,
    results: Any,
    class_names: Optional[Dict[int, str]] = None,
    line_thickness: int = 2,
    font_scale: float = 0.5,
    font_thickness: int = 1
) -> np.ndarray:
    """
    Draw detection results on image.
    
    Args:
        image: Input image (BGR format)
        results: YOLO detection results
        class_names: Dictionary mapping class IDs to names
        line_thickness: Thickness of bounding box lines
        font_scale: Font scale for text
        font_thickness: Font thickness for text
    
    Returns:
        Annotated image
    """
    if results is None:
        return image.copy()
    
    annotated_image = image.copy()
    
    # Check if results have boxes
    if not hasattr(results, 'boxes') or results.boxes is None:
        return annotated_image
    
    boxes = results.boxes
    
    # Process each detection
    for i in range(len(boxes)):
        # Get box coordinates
        if hasattr(boxes, 'xyxy'):
            box = boxes.xyxy[i].cpu().numpy().astype(int)
            x1, y1, x2, y2 = box
        else:
            continue
        
        # Get confidence score
        confidence = 0.0
        if hasattr(boxes, 'conf'):
            confidence = float(boxes.conf[i].cpu().numpy())
        
        # Get class ID and name
        class_id = 0
        if hasattr(boxes, 'cls'):
            class_id = int(boxes.cls[i].cpu().numpy())
        
        class_name = 'Object'
        if class_names and class_id in class_names:
            class_name = class_names[class_id]
        elif hasattr(results, 'names') and class_id in results.names:
            class_name = results.names[class_id]
        
        # Choose color based on class ID
        color = get_color(class_id)
        
        # Draw bounding box
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, line_thickness)
        
        # Draw label
        label = f'{class_name}: {confidence:.2f}'
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
        
        # Draw label background
        cv2.rectangle(
            annotated_image,
            (x1, y1 - label_size[1] - 10),
            (x1 + label_size[0] + 10, y1),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            annotated_image,
            label,
            (x1 + 5, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            font_thickness
        )
    
    return annotated_image


def get_color(class_id: int) -> Tuple[int, int, int]:
    """
    Get color for a specific class ID.
    
    Args:
        class_id: Class identifier
    
    Returns:
        BGR color tuple
    """
    # Predefined colors for different classes
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 128),  # Purple
        (255, 165, 0),  # Orange
        (0, 128, 255),  # Light Blue
        (128, 255, 0),  # Light Green
    ]
    
    return colors[class_id % len(colors)]


def create_video_writer(
    output_path: str,
    fps: int,
    width: int,
    height: int,
    fourcc: str = 'mp4v'
) -> cv2.VideoWriter:
    """
    Create a video writer for saving video output.
    
    Args:
        output_path: Output video file path
        fps: Frames per second
        width: Video width
        height: Video height
        fourcc: Video codec fourcc code
    
    Returns:
        OpenCV VideoWriter object
    """
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Create fourcc code
    fourcc_code = cv2.VideoWriter_fourcc(*fourcc)
    
    # Create video writer
    video_writer = cv2.VideoWriter(
        output_path,
        fourcc_code,
        fps,
        (width, height)
    )
    
    if not video_writer.isOpened():
        raise ValueError(f"Failed to create video writer for: {output_path}")
    
    return video_writer


def draw_segmentation_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.3
) -> np.ndarray:
    """
    Draw segmentation mask on image.
    
    Args:
        image: Input image
        mask: Segmentation mask
        color: Mask color (BGR)
        alpha: Transparency factor
    
    Returns:
        Image with mask overlay
    """
    # Create colored mask
    colored_mask = np.zeros_like(image)
    colored_mask[mask > 0] = color
    
    # Blend with original image
    result = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)
    
    return result


def draw_keypoints(
    image: np.ndarray,
    keypoints: np.ndarray,
    connections: Optional[List[Tuple[int, int]]] = None,
    keypoint_color: Tuple[int, int, int] = (0, 255, 0),
    connection_color: Tuple[int, int, int] = (255, 0, 0),
    keypoint_radius: int = 3,
    line_thickness: int = 2
) -> np.ndarray:
    """
    Draw keypoints and connections on image.
    
    Args:
        image: Input image
        keypoints: Keypoints array (N, 3) with [x, y, visibility]
        connections: List of keypoint pairs to connect
        keypoint_color: Color for keypoints
        connection_color: Color for connections
        keypoint_radius: Radius of keypoint circles
        line_thickness: Thickness of connection lines
    
    Returns:
        Image with keypoints drawn
    """
    result = image.copy()
    
    # Draw connections first (so keypoints appear on top)
    if connections is not None:
        for start_idx, end_idx in connections:
            if (start_idx < len(keypoints) and end_idx < len(keypoints) and
                keypoints[start_idx][2] > 0 and keypoints[end_idx][2] > 0):
                
                start_point = (int(keypoints[start_idx][0]), int(keypoints[start_idx][1]))
                end_point = (int(keypoints[end_idx][0]), int(keypoints[end_idx][1]))
                
                cv2.line(result, start_point, end_point, connection_color, line_thickness)
    
    # Draw keypoints
    for keypoint in keypoints:
        if keypoint[2] > 0:  # Check visibility
            center = (int(keypoint[0]), int(keypoint[1]))
            cv2.circle(result, center, keypoint_radius, keypoint_color, -1)
    
    return result


def create_grid_visualization(
    images: List[np.ndarray],
    titles: Optional[List[str]] = None,
    grid_shape: Optional[Tuple[int, int]] = None,
    padding: int = 10,
    title_height: int = 30
) -> np.ndarray:
    """
    Create a grid visualization from multiple images.
    
    Args:
        images: List of images to arrange in grid
        titles: Optional titles for each image
        grid_shape: Grid shape (rows, cols). If None, auto-calculate
        padding: Padding between images
        title_height: Height reserved for titles
    
    Returns:
        Grid visualization image
    """
    if not images:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Auto-calculate grid shape if not provided
    if grid_shape is None:
        num_images = len(images)
        cols = int(np.ceil(np.sqrt(num_images)))
        rows = int(np.ceil(num_images / cols))
        grid_shape = (rows, cols)
    
    rows, cols = grid_shape
    
    # Get dimensions of first image
    h, w = images[0].shape[:2]
    
    # Create grid canvas
    canvas_height = rows * (h + title_height) + (rows + 1) * padding
    canvas_width = cols * w + (cols + 1) * padding
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
    
    # Place images in grid
    for i, image in enumerate(images):
        if i >= rows * cols:
            break
        
        row = i // cols
        col = i % cols
        
        # Calculate position
        y_start = row * (h + title_height) + (row + 1) * padding
        x_start = col * w + (col + 1) * padding
        
        # Add title if provided
        if titles and i < len(titles):
            cv2.putText(
                canvas,
                titles[i],
                (x_start, y_start + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                1
            )
        
        # Place image
        y_img_start = y_start + title_height
        canvas[y_img_start:y_img_start + h, x_start:x_start + w] = image
    
    return canvas


def save_detection_results(
    results: Any,
    output_path: str,
    format: str = 'txt'
) -> None:
    """
    Save detection results to file.
    
    Args:
        results: YOLO detection results
        output_path: Output file path
        format: Output format ('txt', 'json', 'csv')
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    if format.lower() == 'txt':
        _save_results_txt(results, output_path)
    elif format.lower() == 'json':
        _save_results_json(results, output_path)
    elif format.lower() == 'csv':
        _save_results_csv(results, output_path)
    else:
        raise ValueError(f"Unsupported format: {format}")


def _save_results_txt(results: Any, output_path: str):
    """Save results in YOLO txt format."""
    with open(output_path, 'w') as f:
        if hasattr(results, 'boxes') and results.boxes is not None:
            boxes = results.boxes
            for i in range(len(boxes)):
                if hasattr(boxes, 'cls') and hasattr(boxes, 'conf') and hasattr(boxes, 'xyxy'):
                    class_id = int(boxes.cls[i].cpu().numpy())
                    confidence = float(boxes.conf[i].cpu().numpy())
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    
                    f.write(f"{class_id} {confidence:.6f} {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f}\n")


def _save_results_json(results: Any, output_path: str):
    """Save results in JSON format."""
    import json
    
    detection_data = {
        'detections': []
    }
    
    if hasattr(results, 'boxes') and results.boxes is not None:
        boxes = results.boxes
        for i in range(len(boxes)):
            detection = {}
            
            if hasattr(boxes, 'cls'):
                detection['class_id'] = int(boxes.cls[i].cpu().numpy())
            if hasattr(boxes, 'conf'):
                detection['confidence'] = float(boxes.conf[i].cpu().numpy())
            if hasattr(boxes, 'xyxy'):
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                detection['bbox'] = [float(x1), float(y1), float(x2), float(y2)]
            
            detection_data['detections'].append(detection)
    
    with open(output_path, 'w') as f:
        json.dump(detection_data, f, indent=2)


def _save_results_csv(results: Any, output_path: str):
    """Save results in CSV format."""
    import csv
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['class_id', 'confidence', 'x1', 'y1', 'x2', 'y2'])
        
        if hasattr(results, 'boxes') and results.boxes is not None:
            boxes = results.boxes
            for i in range(len(boxes)):
                row = []
                
                if hasattr(boxes, 'cls'):
                    row.append(int(boxes.cls[i].cpu().numpy()))
                else:
                    row.append(0)
                
                if hasattr(boxes, 'conf'):
                    row.append(float(boxes.conf[i].cpu().numpy()))
                else:
                    row.append(0.0)
                
                if hasattr(boxes, 'xyxy'):
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    row.extend([float(x1), float(y1), float(x2), float(y2)])
                else:
                    row.extend([0.0, 0.0, 0.0, 0.0])
                
                writer.writerow(row)