"""
Data loading utilities for YOLO11 project.

This module provides functions for loading and preprocessing
images, videos, and other data sources.
"""

import cv2
import numpy as np
import torch
from typing import Optional, Union, List, Tuple, Dict, Any
from pathlib import Path
import logging
import os

logger = logging.getLogger(__name__)


def load_image(
    image_path: Union[str, Path],
    target_size: Optional[Tuple[int, int]] = None,
    normalize: bool = False
) -> Optional[np.ndarray]:
    """
    Load and preprocess an image.
    
    Args:
        image_path: Path to image file
        target_size: Target size (width, height) for resizing
        normalize: Whether to normalize pixel values to [0, 1]
    
    Returns:
        Loaded image as numpy array (BGR format) or None if failed
    """
    try:
        # Check if file exists
        if not Path(image_path).exists():
            logger.error(f"Image file not found: {image_path}")
            return None
        
        # Load image
        image = cv2.imread(str(image_path))
        
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return None
        
        # Resize if target size specified
        if target_size is not None:
            image = cv2.resize(image, target_size)
        
        # Normalize if requested
        if normalize:
            image = image.astype(np.float32) / 255.0
        
        return image
        
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        return None


def load_video(
    video_path: Union[str, Path]
) -> Optional[cv2.VideoCapture]:
    """
    Load a video file.
    
    Args:
        video_path: Path to video file
    
    Returns:
        OpenCV VideoCapture object or None if failed
    """
    try:
        # Check if file exists
        if not Path(video_path).exists():
            logger.error(f"Video file not found: {video_path}")
            return None
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return None
        
        return cap
        
    except Exception as e:
        logger.error(f"Error loading video {video_path}: {e}")
        return None


def get_video_info(video_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """
    Get video file information.
    
    Args:
        video_path: Path to video file
    
    Returns:
        Dictionary with video information or None if failed
    """
    cap = load_video(video_path)
    if cap is None:
        return None
    
    try:
        info = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
        }
        
        cap.release()
        return info
        
    except Exception as e:
        logger.error(f"Error getting video info for {video_path}: {e}")
        cap.release()
        return None


def load_image_batch(
    image_paths: List[Union[str, Path]],
    target_size: Optional[Tuple[int, int]] = None,
    normalize: bool = False,
    max_batch_size: int = 32
) -> List[np.ndarray]:
    """
    Load a batch of images.
    
    Args:
        image_paths: List of image file paths
        target_size: Target size for resizing
        normalize: Whether to normalize pixel values
        max_batch_size: Maximum batch size to process at once
    
    Returns:
        List of loaded images
    """
    images = []
    
    for i in range(0, len(image_paths), max_batch_size):
        batch_paths = image_paths[i:i + max_batch_size]
        
        for image_path in batch_paths:
            image = load_image(image_path, target_size, normalize)
            if image is not None:
                images.append(image)
            else:
                logger.warning(f"Skipping failed image: {image_path}")
    
    return images


def create_data_loader(
    data_source: Union[str, Path, List],
    batch_size: int = 1,
    shuffle: bool = False,
    target_size: Optional[Tuple[int, int]] = None
) -> 'DataLoader':
    """
    Create a data loader for images or videos.
    
    Args:
        data_source: Path to image/video file, directory, or list of paths
        batch_size: Batch size for loading
        shuffle: Whether to shuffle the data
        target_size: Target size for resizing
    
    Returns:
        DataLoader instance
    """
    return DataLoader(data_source, batch_size, shuffle, target_size)


class DataLoader:
    """
    Custom data loader for YOLO11 inference.
    """
    
    def __init__(
        self,
        data_source: Union[str, Path, List],
        batch_size: int = 1,
        shuffle: bool = False,
        target_size: Optional[Tuple[int, int]] = None
    ):
        """
        Initialize data loader.
        
        Args:
            data_source: Data source (file, directory, or list)
            batch_size: Batch size
            shuffle: Whether to shuffle data
            target_size: Target size for resizing
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.target_size = target_size
        self.current_index = 0
        
        # Parse data source
        self.file_paths = self._parse_data_source(data_source)
        
        if self.shuffle:
            np.random.shuffle(self.file_paths)
        
        logger.info(f"DataLoader initialized with {len(self.file_paths)} files")
    
    def _parse_data_source(self, data_source: Union[str, Path, List]) -> List[Path]:
        """Parse data source and return list of file paths."""
        if isinstance(data_source, list):
            return [Path(p) for p in data_source]
        
        data_path = Path(data_source)
        
        if data_path.is_file():
            return [data_path]
        elif data_path.is_dir():
            # Supported image and video extensions
            image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
            video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}
            supported_exts = image_exts | video_exts
            
            file_paths = []
            for ext in supported_exts:
                file_paths.extend(data_path.glob(f'*{ext}'))
                file_paths.extend(data_path.glob(f'*{ext.upper()}'))
            
            return sorted(file_paths)
        else:
            logger.error(f"Data source not found: {data_source}")
            return []
    
    def __len__(self) -> int:
        """Return number of batches."""
        return len(self.file_paths) // self.batch_size + (1 if len(self.file_paths) % self.batch_size > 0 else 0)
    
    def __iter__(self):
        """Iterator interface."""
        self.current_index = 0
        return self
    
    def __next__(self) -> List[np.ndarray]:
        """Get next batch."""
        if self.current_index >= len(self.file_paths):
            raise StopIteration
        
        # Get batch file paths
        end_index = min(self.current_index + self.batch_size, len(self.file_paths))
        batch_paths = self.file_paths[self.current_index:end_index]
        
        # Load batch
        batch_data = []
        for file_path in batch_paths:
            if file_path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}:
                # Load image
                image = load_image(file_path, self.target_size)
                if image is not None:
                    batch_data.append(image)
            else:
                # For videos, we'll return the path for now
                # In practice, you might want to extract frames
                batch_data.append(str(file_path))
        
        self.current_index = end_index
        return batch_data
    
    def reset(self):
        """Reset iterator."""
        self.current_index = 0
        if self.shuffle:
            np.random.shuffle(self.file_paths)


def preprocess_image_for_yolo(
    image: np.ndarray,
    input_size: int = 640,
    auto_pad: bool = True
) -> Tuple[torch.Tensor, float, Tuple[int, int]]:
    """
    Preprocess image for YOLO inference.
    
    Args:
        image: Input image (BGR format)
        input_size: Target input size
        auto_pad: Whether to apply padding to maintain aspect ratio
    
    Returns:
        Preprocessed tensor, scale factor, and padding
    """
    # Get original dimensions
    h, w = image.shape[:2]
    
    if auto_pad:
        # Calculate scale to fit image in input_size while maintaining aspect ratio
        scale = min(input_size / h, input_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h))
        
        # Calculate padding
        pad_h = input_size - new_h
        pad_w = input_size - new_w
        pad_top = pad_h // 2
        pad_left = pad_w // 2
        
        # Apply padding
        padded = cv2.copyMakeBorder(
            resized,
            pad_top, pad_h - pad_top,
            pad_left, pad_w - pad_left,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114)  # Gray padding
        )
        
        padding = (pad_left, pad_top)
    else:
        # Simple resize without maintaining aspect ratio
        padded = cv2.resize(image, (input_size, input_size))
        scale = input_size / max(h, w)
        padding = (0, 0)
    
    # Convert to RGB and normalize
    rgb_image = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    normalized = rgb_image.astype(np.float32) / 255.0
    
    # Convert to tensor and rearrange dimensions (HWC -> CHW)
    tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
    
    return tensor, scale, padding


def postprocess_yolo_output(
    predictions: torch.Tensor,
    scale: float,
    padding: Tuple[int, int],
    original_shape: Tuple[int, int]
) -> torch.Tensor:
    """
    Postprocess YOLO model output to original image coordinates.
    
    Args:
        predictions: Model predictions
        scale: Scale factor used in preprocessing
        padding: Padding applied in preprocessing
        original_shape: Original image shape (height, width)
    
    Returns:
        Predictions in original image coordinates
    """
    # Remove padding effect
    predictions[..., 0] -= padding[0]  # x
    predictions[..., 1] -= padding[1]  # y
    predictions[..., 2] -= padding[0]  # x2
    predictions[..., 3] -= padding[1]  # y2
    
    # Scale back to original image size
    predictions[..., :4] /= scale
    
    # Clip to image boundaries
    h, w = original_shape
    predictions[..., 0] = torch.clamp(predictions[..., 0], 0, w)  # x1
    predictions[..., 1] = torch.clamp(predictions[..., 1], 0, h)  # y1
    predictions[..., 2] = torch.clamp(predictions[..., 2], 0, w)  # x2
    predictions[..., 3] = torch.clamp(predictions[..., 3], 0, h)  # y2
    
    return predictions


def save_predictions_to_file(
    predictions: List[Dict],
    output_path: Union[str, Path],
    format: str = 'json'
):
    """
    Save predictions to file.
    
    Args:
        predictions: List of prediction dictionaries
        output_path: Output file path
        format: Output format ('json', 'csv', 'txt')
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format.lower() == 'json':
        import json
        with open(output_path, 'w') as f:
            json.dump(predictions, f, indent=2)
    
    elif format.lower() == 'csv':
        import csv
        if predictions:
            with open(output_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=predictions[0].keys())
                writer.writeheader()
                writer.writerows(predictions)
    
    elif format.lower() == 'txt':
        with open(output_path, 'w') as f:
            for pred in predictions:
                f.write(str(pred) + '\n')
    
    logger.info(f"Predictions saved to: {output_path}")


def create_dataset_config(
    dataset_path: Union[str, Path],
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    class_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Create dataset configuration for YOLO training.
    
    Args:
        dataset_path: Path to dataset directory
        train_split: Training data split ratio
        val_split: Validation data split ratio
        test_split: Test data split ratio
        class_names: List of class names
    
    Returns:
        Dataset configuration dictionary
    """
    dataset_path = Path(dataset_path)
    
    config = {
        'path': str(dataset_path),
        'train': 'train',
        'val': 'val',
        'test': 'test',
        'nc': len(class_names) if class_names else 0,
        'names': class_names or []
    }
    
    return config