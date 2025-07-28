"""
YOLO11 Object Detection Demo

This script demonstrates YOLO11 object detection capabilities with
image, video, and webcam inputs.
"""

import argparse
import cv2
import torch
from pathlib import Path
import logging
import sys
import os

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from core.model import YOLO11Model, YOLO11Factory
from utils.visualization import draw_detections, create_video_writer
from utils.data_loader import load_image, load_video

logger = logging.getLogger(__name__)


class DetectionDemo:
    """
    Demo class for YOLO11 object detection.
    """
    
    def __init__(
        self,
        model_path: str = None,
        model_size: str = 'n',
        device: str = None,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45
    ):
        """
        Initialize detection demo.
        
        Args:
            model_path: Path to model weights (if None, uses pretrained)
            model_size: Model size ('n', 's', 'm', 'l', 'x')
            device: Device to run on ('cpu', 'cuda', 'mps')
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Initialize model
        if model_path:
            self.model = YOLO11Model(model_path=model_path, device=device)
        else:
            self.model = YOLO11Factory.create_detector(size=model_size, device=device)
        
        logger.info(f"Detection demo initialized with model: {self.model}")
    
    def detect_image(
        self,
        image_path: str,
        output_path: str = None,
        show: bool = True,
        save: bool = True
    ) -> dict:
        """
        Detect objects in a single image.
        
        Args:
            image_path: Path to input image
            output_path: Path to save output image
            show: Whether to display the image
            save: Whether to save the image
        
        Returns:
            Detection results
        """
        logger.info(f"Processing image: {image_path}")
        
        # Load image
        image = load_image(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Run detection
        results = self.model.predict(
            image_path,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            show=False,
            save=False
        )
        
        # Process results
        if results and len(results) > 0:
            result = results[0]
            
            # Draw detections
            annotated_image = draw_detections(image, result)
            
            # Display image
            if show:
                cv2.imshow('YOLO11 Detection', annotated_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            # Save image
            if save:
                if output_path is None:
                    output_path = str(Path(image_path).parent / f"detected_{Path(image_path).name}")
                cv2.imwrite(output_path, annotated_image)
                logger.info(f"Detection result saved to: {output_path}")
            
            # Extract detection info
            detection_info = {
                'num_detections': len(result.boxes) if result.boxes else 0,
                'classes_detected': [],
                'confidences': [],
                'boxes': []
            }
            
            if result.boxes:
                for box in result.boxes:
                    if hasattr(box, 'cls') and hasattr(box, 'conf') and hasattr(box, 'xyxy'):
                        class_id = int(box.cls.cpu().numpy())
                        confidence = float(box.conf.cpu().numpy())
                        bbox = box.xyxy.cpu().numpy().tolist()
                        
                        detection_info['classes_detected'].append(class_id)
                        detection_info['confidences'].append(confidence)
                        detection_info['boxes'].append(bbox)
            
            return detection_info
        
        return {'num_detections': 0, 'classes_detected': [], 'confidences': [], 'boxes': []}
    
    def detect_video(
        self,
        video_path: str,
        output_path: str = None,
        show: bool = True,
        save: bool = True
    ) -> dict:
        """
        Detect objects in a video.
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video
            show: Whether to display the video
            save: Whether to save the video
        
        Returns:
            Detection summary
        """
        logger.info(f"Processing video: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create video writer if saving
        video_writer = None
        if save:
            if output_path is None:
                output_path = str(Path(video_path).parent / f"detected_{Path(video_path).name}")
            video_writer = create_video_writer(output_path, fps, width, height)
        
        # Process video frames
        frame_count = 0
        total_detections = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Run detection on frame
                results = self.model.predict(
                    frame,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    show=False,
                    save=False
                )
                
                # Draw detections on a copy of the original frame
                display_frame = frame.copy()
                if results and len(results) > 0:
                    result = results[0]
                    display_frame = draw_detections(frame, result)
                    
                    # Count detections
                    if result.boxes:
                        total_detections += len(result.boxes)
                
                # Display frame
                if show:
                    cv2.imshow('YOLO11 Video Detection', display_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Write frame
                if video_writer:
                    video_writer.write(display_frame)
                
                # Progress update
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    logger.info(f"Processing progress: {progress:.1f}% ({frame_count}/{total_frames})")
        
        finally:
            cap.release()
            if video_writer:
                video_writer.release()
            cv2.destroyAllWindows()
        
        summary = {
            'total_frames': frame_count,
            'total_detections': total_detections,
            'avg_detections_per_frame': total_detections / frame_count if frame_count > 0 else 0,
            'fps': fps,
            'resolution': (width, height)
        }
        
        if save:
            logger.info(f"Detection video saved to: {output_path}")
        
        return summary
    
    def detect_webcam(
        self,
        camera_id: int = 0,
        output_path: str = None,
        save: bool = False
    ):
        """
        Real-time detection from webcam.
        
        Args:
            camera_id: Camera device ID
            output_path: Path to save recording
            save: Whether to save the video stream
        """
        logger.info(f"Starting webcam detection (camera {camera_id})")
        
        # Open webcam
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise ValueError(f"Could not open camera {camera_id}")
        
        # Get webcam properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = 30  # Default fps for webcam recording
        
        # Create video writer if saving
        video_writer = None
        if save and output_path:
            video_writer = create_video_writer(output_path, fps, width, height)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Run detection
                results = self.model.predict(
                    frame,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    show=False,
                    save=False
                )
                
                # Draw detections on a copy of the original frame
                display_frame = frame.copy()
                if results and len(results) > 0:
                    result = results[0]
                    display_frame = draw_detections(frame, result)
                
                # Display frame
                cv2.imshow('YOLO11 Webcam Detection (Press Q to quit)', display_frame)
                
                # Write frame
                if video_writer:
                    video_writer.write(display_frame)
                
                # Check for exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            cap.release()
            if video_writer:
                video_writer.release()
            cv2.destroyAllWindows()
        
        logger.info("Webcam detection stopped")


def main():
    """Main function for detection demo."""
    parser = argparse.ArgumentParser(description='YOLO11 Detection Demo')
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Input path (image, video, or "webcam")')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output path for results')
    parser.add_argument('--model', '-m', type=str, default=None,
                       help='Path to model weights')
    parser.add_argument('--size', '-s', type=str, default='n',
                       choices=['n', 's', 'm', 'l', 'x'],
                       help='Model size')
    parser.add_argument('--device', '-d', type=str, default=None,
                       help='Device to run on (cpu, cuda, mps)')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='IoU threshold for NMS')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display results')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save results')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize demo
    demo = DetectionDemo(
        model_path=args.model,
        model_size=args.size,
        device=args.device,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )
    
    # Run detection based on input type
    try:
        if args.input.lower() == 'webcam':
            demo.detect_webcam(
                camera_id=0,
                output_path=args.output,
                save=not args.no_save
            )
        elif Path(args.input).suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
            summary = demo.detect_video(
                video_path=args.input,
                output_path=args.output,
                show=not args.no_show,
                save=not args.no_save
            )
            logger.info(f"Video detection summary: {summary}")
        else:
            results = demo.detect_image(
                image_path=args.input,
                output_path=args.output,
                show=not args.no_show,
                save=not args.no_save
            )
            logger.info(f"Image detection results: {results}")
    
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())