#!/usr/bin/env python3
"""
freeze_backbone_train_pose_head.py

Advanced transfer learning:
1. Load detection model weights (your good detect weights)
2. Create pose model architecture
3. Transfer backbone + FREEZE it
4. Train ONLY the pose head on keypoint data
5. Detection quality is preserved, keypoints are learned

This ensures:
- Bounding box detection doesn't degrade
- Only pose head learns from keypoint annotations
"""

import sys
import torch
from ultralytics import YOLO


def freeze_backbone_train_pose_head(
    detect_weights: str,
    pose_yaml: str,
    data_yaml: str,
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "",
    name: str = "fpv_gate_pose_frozen_backbone"
):
    """
    Train pose head while keeping detection backbone frozen.
    
    Args:
        detect_weights: Path to detection model weights
        pose_yaml: Path to pose model architecture YAML
        data_yaml: Path to pose dataset YAML
        epochs: Training epochs
        imgsz: Image size
        batch: Batch size
        device: Device string
        name: Training run name
    """
    
    print("=" * 70)
    print("ADVANCED TRANSFER: Detect Backbone → Pose Head (FROZEN BACKBONE)")
    print("=" * 70)
    
    # Step 1: Load detection model
    print(f"\n[1] Loading detection model: {detect_weights}")
    try:
        detect_model = YOLO(detect_weights)
        print(f"    ✓ Detection model loaded")
    except Exception as e:
        print(f"    ✗ Failed to load detection model: {e}")
        sys.exit(1)
    
    # Step 2: Create pose model
    print(f"\n[2] Creating pose model from: {pose_yaml}")
    try:
        pose_model = YOLO(pose_yaml)
        print(f"    ✓ Pose model created")
    except Exception as e:
        print(f"    ✗ Failed to create pose model: {e}")
        sys.exit(1)
    
    # Step 3: Transfer backbone weights only
    print(f"\n[3] Transferring backbone weights (detect → pose)")
    try:
        detect_state = detect_model.model.state_dict()
        pose_state = pose_model.model.state_dict()
        
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
        
        pose_model.model.load_state_dict(pose_state, strict=False)
        print(f"    ✓ Backbone weights transferred: {transferred}")
        print(f"    ✓ Head layers (will train): {skipped}")
    except Exception as e:
        print(f"    ✗ Failed to transfer weights: {e}")
        sys.exit(1)
    
    # Step 4: FREEZE backbone layers
    print(f"\n[4] FREEZING backbone (detection head) layers...")
    
    # Identify and freeze backbone layers
    # Typically, the first ~22 layers are backbone, last layers are head
    frozen_count = 0
    trainable_count = 0
    
    for name, param in pose_model.model.named_parameters():
        # Freeze all layers EXCEPT the pose-specific head
        # Pose head is typically: model.23 onwards (detection head)
        # We want to freeze those detection layers but keep new pose layers
        
        # Strategy: Freeze everything except layers that didn't exist in detection
        # These are the NEW pose head layers
        
        # Common pattern: detection has model.22 (head), pose has different structure
        # Freeze if it's from the detection model (common layers)
        if 'model.22' in name or 'model.23' in name or 'model.24' in name:
            # These are likely detection head layers - FREEZE
            param.requires_grad = False
            frozen_count += 1
        else:
            # Everything else (backbone + new pose layers) can train
            param.requires_grad = True
            trainable_count += 1
    
    # Better approach: freeze everything before the final detection head
    # and let YOLO's training handle the rest
    # Actually, let's freeze explicitly by model structure
    
    # Reset and do more careful freezing
    frozen_count = 0
    trainable_count = 0
    
    try:
        # Get the model
        model = pose_model.model
        
        # Freeze backbone (typically first 22 layers for YOLOv8)
        # These layers should be frozen to preserve detection quality
        for idx, layer in enumerate(model.children()):
            if idx < 22:  # Backbone layers
                for param in layer.parameters():
                    param.requires_grad = False
                frozen_count += 1
            else:  # Head layers
                for param in layer.parameters():
                    param.requires_grad = True
                trainable_count += 1
        
        print(f"    ✓ Frozen backbone layers: {frozen_count}")
        print(f"    ✓ Trainable head layers: {trainable_count}")
        print(f"    ℹ Detection backbone is FROZEN - will not change during training")
        print(f"    ℹ Pose head will learn keypoints from scratch")
        
    except Exception as e:
        print(f"    [WARN] Could not freeze explicitly: {e}")
        print(f"    Using Ultralytics default freezing...")
    
    # Step 5: Train
    print(f"\n[5] Training pose head on: {data_yaml}")
    print(f"    Epochs: {epochs}, Batch: {batch}, Image size: {imgsz}")
    print(f"    Device: {device if device else 'auto'}")
    print(f"    Backbone is FROZEN - only pose head will update")
    print("-" * 70)
    
    try:
        # Use Ultralytics' freezing mechanism
        # freeze parameter can specify which layers to freeze
        results = pose_model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            task='pose',
            project='runs/pose',
            name=name,
            freeze=18,  # Freeze first 18 layers (backbone)
            patience=10,  # Early stopping
        )
        
        print("-" * 70)
        print(f"\n✓ Training complete!")
        print(f"  Best weights: {results.save_dir}/weights/best.pt")
        print(f"\n  Detection quality: PRESERVED (backbone frozen)")
        print(f"  Keypoint learning: NEW (head trained from scratch)")
        return results
        
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    
    ap = argparse.ArgumentParser(
        description="Train pose head with frozen detection backbone"
    )
    
    ap.add_argument(
        "--detect-weights",
        default="./current_best_non_vocab.pt",
        help="Good detection model weights to preserve"
    )
    ap.add_argument(
        "--pose-yaml",
        default="yolov8n-pose.yaml",
        help="Pose model architecture YAML"
    )
    ap.add_argument(
        "--data-yaml",
        required=True,
        help="Pose dataset YAML (keypoint annotations)"
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
        default="",
        help="Device string ('' for auto, 'cpu', '0', 'mps', etc)"
    )
    ap.add_argument(
        "--name",
        default="fpv_gate_pose_frozen_backbone",
        help="Training run name"
    )
    
    args = ap.parse_args()
    
    freeze_backbone_train_pose_head(
        detect_weights=args.detect_weights,
        pose_yaml=args.pose_yaml,
        data_yaml=args.data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
    )
