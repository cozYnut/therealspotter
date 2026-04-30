# Keypoint/Pose Support Added to pipeline_sanity_train_eval.py

## Summary
Successfully added complete support for keypoint/pose detection across all three stages of the pipeline: sanity check, training, and evaluation.

## Changes Made

### 1. **Sanity Check Validation** (Lines 95-100)
- **Before**: Only validated 5-column (BB) and 6-column (OBB) labels
- **After**: Now supports:
  - 5 columns: Standard bounding boxes (detect task)
  - 6 columns: Oriented bounding boxes with rotation (obb task)  
  - 5 + keypoints×3 columns: Pose/keypoint format (pose task)
- Validates that pose labels have the correct format: 5 base columns + multiples of 3 for keypoint data

**Error message updated** to: `"wrong column count (expected 5, 6, or 5+keypoints×3)"`

### 2. **Automatic Model Selection** (Lines 175-181)
```python
if task == "obb":
    base_model = "yolov8m-obb.pt"
elif task == "pose":
    base_model = "yolov8n-pose.pt"
else:
    base_model = "yolov8n.pt"
```
- Automatically selects the correct base model based on the task parameter
- Detect → yolov8n.pt (lightweight detection)
- OBB → yolov8m-obb.pt (medium oriented boxes)
- Pose → yolov8n-pose.pt (lightweight keypoint detection)

### 3. **Task Parameter in Training** (Line 195)
```python
results = model.train(
    ...
    task=task,
)
```
- Added `task=task` parameter to `model.train()` call
- Ensures Ultralytics YOLO trainer uses correct task-specific parameters

### 4. **Project Folder Organization** (Line 336)
```python
project = args.project if args.project else (
    "runs/obb" if args.task == "obb" 
    else "runs/pose" if args.task == "pose" 
    else "runs/detect"
)
```
- Automatically organizes training outputs by task type
- Detect training → `runs/detect/`
- OBB training → `runs/obb/`
- Pose training → `runs/pose/`
- User can override with `--project` flag

### 5. **Argument Parser Update** (Line 318)
```python
ap.add_argument("--task", default="detect", 
               choices=["detect", "obb", "pose"], 
               help="Task type: detect, obb, or pose (keypoint)")
```
- Added "pose" as a valid task choice
- Updated help text to clarify keypoint support

### 6. **Project Argument Default** (Line 344)
```python
ap.add_argument("--project", default="", 
               help="Training output base folder (auto by task if empty)")
```
- Changed default from `"runs/detect"` to `""` (empty string)
- Empty string triggers automatic task-based folder selection
- Users can still specify custom `--project` path if desired

### 7. **Evaluation Function Signature** (Line 218)
```python
def run_eval(weights: str, data_yaml: str, split: str, 
            imgsz: int, batch: int, device: str, 
            task: str = "detect"):
```
- Added `task: str = "detect"` parameter to function signature
- Default is "detect" for backward compatibility

### 8. **Task Parameter in Evaluation** (Line 226)
```python
metrics = model.val(
    ...
    task=task,
)
```
- Added `task=task` parameter to `model.val()` call
- Ensures validation uses correct task-specific metrics

### 9. **Eval Call Updated** (Line 384)
```python
run_eval(
    weights=eval_weights,
    data_yaml=data_yaml,
    split=args.eval_split,
    imgsz=args.imgsz,
    batch=args.batch,
    device=args.device,
    task=args.task,  # ← Added
)
```
- Passes `task=args.task` to evaluation function

## Usage Examples

### Train a pose detection model:
```bash
python3 pipeline_sanity_train_eval.py \
  --data-root ./dataset \
  --data-yaml ./fpv_gate.yaml \
  --nc 4 \
  --task pose \
  --train \
  --epochs 100 \
  --batch 16
```

### Evaluate a pose model:
```bash
python3 pipeline_sanity_train_eval.py \
  --data-root ./dataset \
  --data-yaml ./fpv_gate.yaml \
  --nc 4 \
  --task pose \
  --weights ./runs/pose/fpv_gate_train/weights/best.pt \
  --eval-split test
```

### Train OBB with custom project folder:
```bash
python3 pipeline_sanity_train_eval.py \
  --data-root ./dataset \
  --data-yaml ./fpv_gate.yaml \
  --nc 4 \
  --task obb \
  --train \
  --project ./my_custom_runs
```

## Label Format Support

| Task   | Format | Columns | Example |
|--------|--------|---------|---------|
| detect | BB     | 5       | `0 0.5 0.5 0.3 0.4` |
| obb    | OBB    | 6       | `0 0.5 0.5 0.3 0.4 45.5` |
| pose   | Pose   | 5+kp×3  | `0 0.5 0.5 0.3 0.4 x1 y1 c1 x2 y2 c2 ...` |

Where for pose:
- First 5 columns: class_id, center_x, center_y, width, height
- Remaining columns in groups of 3: x, y, confidence for each keypoint
- COCO pose (17 keypoints) = 5 + (17×3) = 56 columns total

## Backward Compatibility

All changes are backward compatible:
- Default task is "detect" (original behavior)
- Existing scripts without `--task pose` flag will work as before
- Project folder auto-selection has sensible defaults matching original behavior

## Testing the Implementation

The sanity check will now properly validate pose format labels during the first stage. If you have pose-annotated data, the validation should pass and proceed to training with the pose model automatically selected.
