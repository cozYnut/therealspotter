#!/usr/bin/env python3
"""
Transfer detect weights to pose model.

This script implements the hybrid approach:
1. Load detection model weights (current_best_non_vocab.pt)
2. Create pose model architecture (yolov8n-pose.yaml)
3. Load detect backbone into pose model (architecture mismatch allowed)
4. Train on pose dataset

This preserves your detection knowledge while adapting to keypoint task.
"""

import sys
from ultralytics import YOLO

def transfer_detect_to_pose(
    detect_weights: str,
    pose_yaml: str,
    data_yaml: str,
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "",
    name: str = "fpv_gate_pose_transfer"
):
    """
    Transfer learn from detection to pose task.
    
    Args:
        detect_weights: Path to detection model weights
        pose_yaml: Path to pose model architecture YAML
        data_yaml: Path to pose dataset YAML
        epochs: Number of training epochs
        imgsz: Image size
        batch: Batch size
        device: Device string ('' for auto, 'cpu', '0', 'mps', etc)
        name: Training run name
    """
    
    print("=" * 60)
    print("TRANSFER LEARNING: Detect → Pose")
    print("=" * 60)
    
    # Step 1: Load detection model
    print(f"\n[1] Loading detection model: {detect_weights}")
    try:
        detect_model = YOLO(detect_weights)
        print(f"    ✓ Detection model loaded")
    except Exception as e:
        print(f"    ✗ Failed to load detection model: {e}")
        sys.exit(1)
    
    # Step 2: Create pose model with proper architecture
    print(f"\n[2] Creating pose model from: {pose_yaml}")
    try:
        pose_model = YOLO(pose_yaml)
        print(f"    ✓ Pose model created")
    except Exception as e:
        print(f"    ✗ Failed to create pose model: {e}")
        sys.exit(1)
    
    # Step 3: Transfer backbone weights (skip incompatible head)
    print(f"\n[3] Transferring backbone weights (detect → pose)")
    print(f"    Note: Pose head will initialize randomly (different architecture)")
    try:
        # Get state dicts
        detect_state = detect_model.model.state_dict()
        pose_state = pose_model.model.state_dict()
        
        # Transfer only backbone weights (skip detection head)
        # Detection head is typically the last few layers
        # Keep only common backbone layers
        transferred = 0
        skipped = 0
        
        for key, value in detect_state.items():
            if key in pose_state:
                try:
                    if pose_state[key].shape == value.shape:
                        pose_state[key] = value
                        transferred += 1
                    else:
                        skipped += 1
                except Exception:
                    skipped += 1
            else:
                skipped += 1
        
        # Load the merged state dict
        pose_model.model.load_state_dict(pose_state, strict=False)
        
        print(f"    ✓ Weights transferred successfully")
        print(f"    ✓ Backbone layers transferred: {transferred}")
        print(f"    ✓ Head layers skipped (will train from scratch): {skipped}")
        print(f"    ℹ Detection backbone reused, pose head initializes randomly")
    except Exception as e:
        print(f"    ✗ Failed to transfer weights: {e}")
        sys.exit(1)
    
    # Step 4: Train on pose data
    print(f"\n[4] Training pose model on: {data_yaml}")
    print(f"    Epochs: {epochs}, Batch: {batch}, Image size: {imgsz}")
    print(f"    Device: {device if device else 'auto'}")
    print("-" * 60)
    
    try:
        results = pose_model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            task='pose',
            project='runs/pose',
            name=name,
        )
        
        print("-" * 60)
        print(f"\n✓ Training complete!")
        print(f"  Best weights: {results.save_dir}/weights/best.pt")
        return results
        
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    
    ap = argparse.ArgumentParser(
        description="Transfer detection weights to pose model"
    )
    
    ap.add_argument(
        "--detect-weights",
        default="./current_best_non_vocab.pt",
        help="Detection model weights to transfer from"
    )
    ap.add_argument(
        "--pose-yaml",
        default="yolov8n-pose.yaml",
        help="Pose model architecture YAML"
    )
    ap.add_argument(
        "--data-yaml",
        required=True,
        help="Pose dataset YAML (e.g., fpv_gate_pose.yaml)"
    )
    ap.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Training epochs"
    )
    ap.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size"
    )
    ap.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size"
    )
    ap.add_argument(
        "--device",
        default="mps",
        help="Device string ('' for auto, 'cpu', '0', 'mps', etc)"
    )
    ap.add_argument(
        "--name",
        default="fpv_gate_pose_transfer",
        help="Training run name"
    )
    
    args = ap.parse_args()
    
    transfer_detect_to_pose(
        detect_weights=args.detect_weights,
        pose_yaml=args.pose_yaml,
        data_yaml=args.data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
    )
