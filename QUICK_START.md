# Quick Start: Multi-Model Support

## One-Liner Commands

### List Available Models
```bash
python3 lazy_spotter.py --list-models
```

### Run Detection with Different Models

**Standard Bounding Box (BB)**:
```bash
python3 lazy_spotter.py --model fpv_gate_bb --video video.mp4 --pass-enable
```

**Oriented Bounding Box (OBB)** - rotated boxes with angle:
```bash
python3 lazy_spotter.py --model fpv_gate_obb --video video.mp4
```

**Pose Estimation** - keypoints:
```bash
python3 lazy_spotter.py --model yolov8m-pose --video video.mp4
```

### Inspect Video with Model

```bash
# Any model type - auto-detects visualization
python3 inspect_gate_model_video.py --model fpv_gate_bb --video video.mp4
```

### Generate Pseudo-Labels

**BB format (5 columns)**:
```bash
python3 fpv_pseudo_label.py --video_dir ./videos --out_dir ./dataset --model yolov8n
```

**OBB format (6 columns)**:
```bash
python3 fpv_pseudo_label.py --video_dir ./videos --out_dir ./dataset_obb --model yolov8m-obb
```

## Model Registry Keys

**Available Now**:
```
fpv_gate_bb           - Your FPV gate detector (BB format)
yolov8n              - YOLOv8 Nano general detector
yolov8m              - YOLOv8 Medium general detector
yolov8m-obb          - YOLOv8 Medium with rotation support
```

**Available for Download/Training**:
```
fpv_gate_obb         - Your FPV detector with rotation (needs training)
yolov8n-pose         - YOLOv8 Nano pose estimation (17 keypoints)
yolov8m-pose         - YOLOv8 Medium pose estimation (17 keypoints)
```

## What's Different in Each Model Type

| Aspect | BB | OBB | Pose |
|--------|----|----|------|
| **Output Columns** | 5 | 6 | 5 + keypoints×3 |
| **Rotation** | No | Yes (degrees) | No |
| **Keypoints** | No | No | Yes (17 for COCO) |
| **Drawing** | Rectangle | Rotated box | Box + circles |

## Label Formats

**BB (Standard, 5 columns)**:
```
0 0.5 0.5 0.2 0.3
```
(class, x_center, y_center, width, height)

**OBB (Rotated, 6 columns)**:
```
0 0.5 0.5 0.2 0.3 45.0
```
(class, x_center, y_center, width, height, rotation_degrees)

**Pose (Keypoints)**:
```
0 0.5 0.5 0.2 0.3 0.1 0.1 0.9 0.2 0.2 0.8 ...
```
(class, bbox_center_x, bbox_center_y, bbox_width, bbox_height, then x, y, confidence for each keypoint)

## Adding a Custom Model

Edit `model_config.py` and add to `ModelRegistry.MODELS`:

```python
"my_model": ModelConfig(
    model_type=ModelType.BB,  # or OBB, KEYPOINTS
    name="My Custom Detector",
    description="What it detects",
    model_path="path/to/weights.pt",
    task="detect",  # or "obb", "pose"
    has_rotation=False,  # True for OBB
    has_keypoints=False,  # True for pose
    default_conf=0.25,
    default_maxdet=50,
),
```

Then use it:
```bash
python3 lazy_spotter.py --model my_model --video video.mp4
```

## Key Features

✅ **Auto Detection** - Automatically detects model type and formats output correctly  
✅ **Backward Compatible** - Old `--det-model` arguments still work  
✅ **Easy Switching** - Change models with just `--model` flag  
✅ **Consistent Interface** - All tools use same model system  
✅ **Label Format Aware** - Generates correct label columns for each model type  

## Common Issues

**Model not found?**
```bash
python3 lazy_spotter.py --list-models
```
Check the exact key name and make sure it's in the list.

**Wrong output format?**
The tool auto-detects. If still wrong, check that your model file actually matches the registered model type in `model_config.py`.

**Legacy arguments not working?**
Still supported but show deprecation warning. Update to use `--model` for new code.

## Next Steps

1. **Try it**: Run `python3 lazy_spotter.py --list-models`
2. **Pick a model**: Choose from available models
3. **Use it**: `python3 lazy_spotter.py --model fpv_gate_bb --video video.mp4`
4. **Inspect results**: `python3 inspect_gate_model_video.py --model fpv_gate_bb --video video.mp4`
5. **Generate labels**: `python3 fpv_pseudo_label.py --model fpv_gate_bb --video_dir videos --out_dir labels`

That's it! The system handles the rest.
