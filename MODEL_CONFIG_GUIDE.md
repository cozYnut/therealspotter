# Multi-Model Type Support Guide

This guide explains how to use the new model configuration system that supports 3 YOLO model types: **Bounding Box (BB)**, **Oriented Bounding Box (OBB)**, and **Key-Points (Pose)**.

## Overview

The new system centralizes model configuration in `model_config.py` and provides a unified interface across all tools.

### Model Types

1. **BB (Bounding Box)** - Standard YOLO detection
   - Output: 5 columns (class_id, x_center, y_center, width, height)
   - Task: `detect`
   - Models: `fpv_gate_bb`, `yolov8n`, `yolov8m`

2. **OBB (Oriented Bounding Box)** - Rotated detection
   - Output: 6 columns (class_id, x_center, y_center, width, height, rotation_angle)
   - Task: `obb`
   - Models: `fpv_gate_obb`, `yolov8m-obb`

3. **Key-Points (Pose)** - Pose/landmark detection
   - Output: 5 + (num_keypoints × 3) columns (bbox + x, y, confidence per keypoint)
   - Task: `pose`
   - Models: `yolov8n-pose`, `yolov8m-pose`

## Files Modified

### New Files
- **`model_config.py`** - Central model registry and utilities

### Updated Files
- **`lazy_spotter.py`** - Main detection pipeline
- **`inspect_gate_model_video.py`** - Video model inspection tool
- **`fpv_pseudo_label.py`** - Pseudo-label generation tool
- **`pipeline_sanity_train_eval.py`** - Training and validation script

## Using the System

### Listing Available Models

All tools now support `--list-models`:

```bash
# List models in any tool
python3 lazy_spotter.py --list-models
python3 inspect_gate_model_video.py --list-models
python3 fpv_pseudo_label.py --list-models
```

Output:
```
Available Models:
  fpv_gate_bb: FPV Gate BB - FPV gate detection with standard bounding boxes
  yolov8n: YOLOv8 Nano - YOLOv8 Nano pretrained model for general detection
  yolov8m: YOLOv8 Medium - YOLOv8 Medium pretrained model for general detection
  yolov8m-obb: YOLOv8 Medium OBB - YOLOv8 Medium pretrained OBB model
  yolov8n-pose: YOLOv8 Nano Pose - YOLOv8 Nano for pose/keypoint detection
  yolov8m-pose: YOLOv8 Medium Pose - YOLOv8 Medium for pose/keypoint detection
```

### lazy_spotter.py

#### New Usage

```bash
# Use registry key (recommended)
python3 lazy_spotter.py \
  --model fpv_gate_bb \
  --video "path/to/video.mp4" \
  --pass-enable

# With OBB model
python3 lazy_spotter.py \
  --model fpv_gate_obb \
  --video "path/to/video.mp4"

# With pose model
python3 lazy_spotter.py \
  --model yolov8m-pose \
  --video "path/to/video.mp4"
```

#### Legacy Usage (Still Supported)

```bash
# Direct path (backward compatible)
python3 lazy_spotter.py \
  --det-model "runs/detect/fpv_gate_train/weights/best.pt" \
  --video "path/to/video.mp4"
```

#### Key Changes
- New: `--model` flag accepts registry keys or direct paths
- New: `--list-models` shows available models
- Legacy: `--det-model` still works but shows deprecation warning
- Auto-detects output format (OBB vs BB) based on model type

### inspect_gate_model_video.py

#### New Usage

```bash
# Use registry key
python3 inspect_gate_model_video.py \
  --model fpv_gate_bb \
  --video "path/to/video.mp4" \
  --conf 0.25

# With OBB model (auto-detects and draws rotated boxes)
python3 inspect_gate_model_video.py \
  --model yolov8m-obb \
  --video "path/to/video.mp4"

# With pose model (draws boxes + keypoints)
python3 inspect_gate_model_video.py \
  --model yolov8m-pose \
  --video "path/to/video.mp4"
```

#### Key Changes
- Auto-detects model type from model file
- OBB models: draws rotated rectangles using `cv2.RotatedRect()`
- Keypoint models: draws bounding boxes + keypoint circles
- BB models: draws standard rectangles

### fpv_pseudo_label.py

#### New Usage

```bash
# Generate BB labels (5 columns)
python3 fpv_pseudo_label.py \
  --video_dir "./videos" \
  --out_dir "./dataset_labels" \
  --model yolov8n

# Generate OBB labels (6 columns)
python3 fpv_pseudo_label.py \
  --video_dir "./videos" \
  --out_dir "./dataset_labels_obb" \
  --model yolov8m-obb

# Generate with confidence override
python3 fpv_pseudo_label.py \
  --video_dir "./videos" \
  --out_dir "./dataset_labels" \
  --model fpv_gate_bb \
  --conf 0.3 \
  --maxdet 100
```

#### Key Changes
- New: `--model` flag accepts registry keys
- New: `--list-models` shows available models
- Automatically generates correct label format based on model type
- Legacy: `--yolo_model` still works but shows deprecation warning

### pipeline_sanity_train_eval.py

No changes needed - already supports multi-format datasets:

```bash
# Works with both BB and OBB format labels
python3 pipeline_sanity_train_eval.py \
  --data-root ./datastuff/fpv_dataset \
  --data-yaml ./fpv_gate.yaml \
  --nc 4 \
  --train \
  --epochs 50
```

## Adding New Models

To register a new model, edit `model_config.py` and add an entry to `ModelRegistry.MODELS`:

```python
ModelRegistry.MODELS["my_custom_model"] = ModelConfig(
    model_type=ModelType.BB,  # or OBB, KEYPOINTS
    name="My Custom Model",
    description="Description of what it detects",
    model_path="path/to/model.pt",
    task="detect",  # or "obb", "pose"
    has_rotation=False,
    has_keypoints=False,
    default_conf=0.25,
    default_maxdet=50,
)
```

Or programmatically:

```python
from model_config import ModelRegistry, ModelConfig, ModelType

config = ModelConfig(...)
ModelRegistry.add_model("my_model", config)
```

## Label Format Details

### Standard BB (5 columns)
```
class_id x_center y_center width height
0 0.5 0.5 0.2 0.3
1 0.7 0.6 0.15 0.25
```

### OBB (6 columns)
```
class_id x_center y_center width height rotation_angle
0 0.5 0.5 0.2 0.3 45.0
1 0.7 0.6 0.15 0.25 90.0
```

Rotation angle is in degrees (0-360), counterclockwise positive.

### Keypoints/Pose (5 + keypoints × 3)
```
class_id x_center y_center width height kp1_x kp1_y kp1_conf kp2_x kp2_y kp2_conf ...
```

## Detection Output Auto-Detection

All tools automatically detect the model type from the results:

1. **OBB Detection**: Checks for `res.obb` attribute and draws rotated rectangles
2. **Keypoint Detection**: Checks for `res.keypoints` attribute and draws both boxes and keypoints
3. **BB Detection**: Falls back to `res.boxes` for standard rectangles

This means if you load an OBB model in `inspect_gate_model_video.py`, it automatically detects and renders OBB results correctly without any additional flags.

## Troubleshooting

### "Model file not found" Error

```
[ERROR] Model file not found: runs/detect/fpv_gate_train/weights/best.pt
```

**Solution**: Make sure the model path in `model_config.py` is correct relative to where you run the script.

### Model not in registry

```
[ERROR] Model 'my_model' not found in registry.
Available models:
  fpv_gate_bb: ...
```

**Solution**: Check the exact model key with `--list-models` or add it to `ModelRegistry.MODELS`.

### Label format mismatch

If you get "labels require X columns, Y columns detected", the model type doesn't match your labels:

- BB model expects 5 columns
- OBB model expects 6 columns
- Pose model expects 5 + (keypoints × 3) columns

**Solution**: Use matching label format or convert labels using the provided utilities.

## Examples

### Complete Workflow: Train OBB Model

```bash
# 1. Generate pseudo-labels with OBB format
python3 fpv_pseudo_label.py \
  --video_dir ./videos \
  --out_dir ./dataset \
  --model yolov8m-obb

# 2. Validate dataset
python3 pipeline_sanity_train_eval.py \
  --data-root ./dataset \
  --data-yaml ./fpv_gate.yaml \
  --nc 4

# 3. Train OBB model
python3 pipeline_sanity_train_eval.py \
  --data-root ./dataset \
  --data-yaml ./fpv_gate.yaml \
  --nc 4 \
  --task obb \
  --train \
  --epochs 50

# 4. Inspect results
python3 inspect_gate_model_video.py \
  --model fpv_gate_obb \
  --video "./test_video.mp4"

# 5. Run detection pipeline
python3 lazy_spotter.py \
  --model fpv_gate_obb \
  --video "./test_video.mp4" \
  --pass-enable
```

### Switch Between Model Types

```bash
# BB detections
python3 lazy_spotter.py --model fpv_gate_bb --video video.mp4

# OBB detections  
python3 lazy_spotter.py --model fpv_gate_obb --video video.mp4

# Pose estimation
python3 lazy_spotter.py --model yolov8m-pose --video video.mp4
```

All three will work seamlessly with automatic output format detection.

## Architecture

The system is built on these core components:

- **`ModelType` Enum**: Defines the 3 model types
- **`ModelConfig` Dataclass**: Stores configuration for a single model
- **`ModelRegistry` Class**: Central registry managing all models
- **Detection utility functions**:
  - `detection_to_yolo_line()`: Converts detections to label format
  - `get_label_columns_count()`: Calculates output columns needed

Each tool uses these components to:
1. Load the appropriate model
2. Run detection
3. Automatically format output based on model type
4. Visualize results correctly (rotated boxes for OBB, keypoints for pose, etc.)
