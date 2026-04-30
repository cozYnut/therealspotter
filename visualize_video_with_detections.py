#!/usr/bin/env python3
"""
visualize_video_with_detections.py

Run a YOLO model on a video file and visualize detections in real-time.
Supports detect, obb, and pose tasks.
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# --------------------------------------------------
# Class names and colors
# --------------------------------------------------
CLASS_NAMES = [
    "square",
    "arch",
    "circle",
    "flagpole",
]

COLORS = [
    (0, 255, 0),     # square - green
    (0, 255, 255),   # arch   - yellow
    (255, 0, 0),     # circle - blue
    (255, 0, 255),   # flag   - magenta
]


def draw_detections(img, results, task='detect'):
    """
    Draw detections on image based on task type.
    
    Args:
        img: Image to draw on
        results: YOLO results object
        task: 'detect', 'obb', or 'pose'
    """
    h, w = img.shape[:2]
    
    if results.boxes is None or len(results.boxes) == 0:
        return img
    
    # Get results
    boxes = results.boxes.cpu().numpy()
    
    for idx, box in enumerate(boxes):
        class_id = int(box.cls[0])
        conf = float(box.conf[0])
        color = COLORS[class_id % len(COLORS)]
        class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"id{class_id}"
        
        if task == 'detect':
            # Standard bounding box: [x1, y1, x2, y2]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name} {conf:.2f}"
            cv2.putText(img, label, (x1, max(20, y1 - 5)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        
        elif task == 'obb':
            # OBB has xywh + rotation angle
            if hasattr(results, 'obb') and results.obb is not None:
                # OBB format with rotation
                x, y, w, h = box.xywh[0]
                angle = float(box.data[4]) if box.data.shape[1] > 4 else 0
                
                x_center, y_center = int(x), int(y)
                width, height = int(w), int(h)
                
                # Draw rotated box
                angle_rad = np.radians(angle)
                cos_a = np.cos(angle_rad)
                sin_a = np.sin(angle_rad)
                
                hw = width / 2
                hh = height / 2
                
                corners_local = np.array([
                    [-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]
                ])
                
                rotation_matrix = np.array([
                    [cos_a, -sin_a],
                    [sin_a, cos_a]
                ])
                
                corners_rotated = corners_local @ rotation_matrix.T
                corners = (corners_rotated + np.array([x_center, y_center])).astype(np.int32)
                cv2.polylines(img, [corners], isClosed=True, color=color, thickness=2)
                
                label = f"{class_name} {conf:.2f}"
                cv2.putText(img, label, (x_center - 30, max(20, y_center - 20)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
            else:
                # Fallback: draw standard box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                label = f"{class_name} {conf:.2f}"
                cv2.putText(img, label, (x1, max(20, y1 - 5)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        
        elif task == 'pose':
            # Pose: keypoints
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Draw keypoints if available
            if results.keypoints is not None and len(results.keypoints) > idx:
                kpts = results.keypoints[idx].xy[0]  # Shape: (num_keypoints, 2)
                for kpt in kpts:
                    x, y = int(kpt[0]), int(kpt[1])
                    if x > 0 and y > 0:  # Valid keypoint
                        cv2.circle(img, (x, y), 4, (0, 255, 0), -1)
                        cv2.circle(img, (x, y), 4, color, 1)
            
            label = f"{class_name} {conf:.2f}"
            cv2.putText(img, label, (x1, max(20, y1 - 5)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    
    return img


def visualize_video(video_path, model_path, task='detect', conf_thresh=0.5, device=''):
    """
    Run YOLO model on video and visualize detections.
    
    Args:
        video_path: Path to video file
        model_path: Path to YOLO model weights
        task: 'detect', 'obb', or 'pose'
        conf_thresh: Confidence threshold
        device: Device string ('cpu', '0', 'mps', etc)
    """
    
    # Validate inputs
    if not os.path.exists(video_path):
        print(f"ERROR: Video not found: {video_path}")
        sys.exit(1)
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found: {model_path}")
        sys.exit(1)
    
    # Load model
    print(f"Loading model: {model_path}")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        sys.exit(1)
    
    # Open video
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"ERROR: Failed to open video: {video_path}")
        sys.exit(1)
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video info: {w}x{h} @ {fps:.1f} FPS, {total_frames} frames")
    print(f"Task: {task}")
    print(f"Confidence threshold: {conf_thresh}")
    print()
    
    # Create window
    window_name = f"Video Detection - {task.upper()}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    frame_count = 0
    paused = False
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("\nEnd of video.")
                break
            frame_count += 1
        
        # Run inference
        try:
            results = model(frame, conf=conf_thresh, task=task, verbose=False)[0]
        except Exception as e:
            print(f"ERROR during inference: {e}")
            results = None
        
        # Draw detections
        if results is not None:
            frame = draw_detections(frame, results, task=task)
        
        # Add info overlay
        info_lines = [
            f"Frame: {frame_count}/{total_frames}",
            f"Task: {task.upper()}",
            f"Model: {os.path.basename(model_path)}",
        ]
        
        y = 25
        for line in info_lines:
            cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (0, 255, 0), 1, cv2.LINE_AA)
            y += 20
        
        # Help text
        help_text = "Space: pause  [←/→]: frame  [q]: quit"
        cv2.putText(frame, help_text, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Display
        cv2.imshow(window_name, frame)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27:  # q or ESC
            print("Quit requested.")
            break
        elif key == ord(' '):  # Space
            paused = not paused
            state = "PAUSED" if paused else "PLAYING"
            print(f"[{state}] Frame {frame_count}/{total_frames}")
        elif key == ord('.'):  # . for next frame (paused)
            if paused:
                # Already handled in main loop
                pass
        elif key == 82 or key == ord('<'):  # Up arrow or <
            if paused:
                frame_count = max(0, frame_count - 2)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
                print(f"Seeking to frame {frame_count}")
        elif key == 84 or key == ord('>'):  # Down arrow or >
            # Frame will auto-advance
            pass
    
    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    import argparse
    
    ap = argparse.ArgumentParser(
        description="Visualize YOLO detections on video"
    )
    
    ap.add_argument(
        "video",
        help="Path to video file"
    )
    ap.add_argument(
        "--model",
        default="current_best_non_vocab.pt",
        help="Path to YOLO model weights"
    )
    ap.add_argument(
        "--task",
        default="detect",
        choices=["detect", "obb", "pose"],
        help="Task type: detect, obb, or pose"
    )
    ap.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Confidence threshold (0-1)"
    )
    ap.add_argument(
        "--device",
        default="",
        help="Device string: '' (auto), 'cpu', '0', 'mps', etc"
    )
    
    args = ap.parse_args()
    
    visualize_video(
        video_path=args.video,
        model_path=args.model,
        task=args.task,
        conf_thresh=args.conf,
        device=args.device
    )
