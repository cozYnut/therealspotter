import argparse
import time
from typing import Dict, Tuple, List

import cv2
from ultralytics import YOLO

# Model configuration system
from model_config import ModelRegistry, ModelType, ModelConfig


# --- Default class-name mapping (edit if your training used different names) ---
# If your .pt has embedded names, we’ll prefer them automatically.
DEFAULT_CLASS_NAMES = {
    0: "square",
    1: "arch",
    2: "circle",
    3: "flagpole",
}

# Optional consistent colors for readability
DEFAULT_COLORS = {
    "square": (255, 0, 0),      # blue
    "circle": (0, 255, 0),      # green
    "arch": (0, 255, 255),      # yellow
    "flagpole": (255, 0, 255),  # magenta
    "unknown": (0, 165, 255),   # orange
}


def get_class_names(model: YOLO, override: Dict[int, str] | None) -> Dict[int, str]:
    # Prefer override if provided, else model.names if present, else DEFAULT_CLASS_NAMES
    if override:
        return override
    names = getattr(model, "names", None)
    if isinstance(names, dict) and names:
        # ultralytics usually stores {id: "name"}
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list) and names:
        return {i: str(n) for i, n in enumerate(names)}
    return dict(DEFAULT_CLASS_NAMES)


def draw_legend(frame, class_names: Dict[int, str], colors: Dict[str, Tuple[int, int, int]]):
    x, y = 10, 20
    cv2.putText(frame, "Legend:", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2, cv2.LINE_AA)
    y += 18
    for cid in sorted(class_names.keys()):
        name = class_names[cid]
        c = colors.get(name, colors["unknown"])
        cv2.rectangle(frame, (x, y - 10), (x + 14, y + 4), c, -1)
        cv2.putText(frame, f"{cid}: {name}", (x + 20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 2, cv2.LINE_AA)
        y += 18


def run_detection_on_video(model_path, video_path, conf=0.25):
    """Run OBB detection on video"""
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run OBB inference
        results = model(frame, conf=conf)
        
        for r in results:
            if r.obb is not None:
                # Draw OBB boxes
                for box_data in r.obb.data:
                    x, y, w, h, rotation = box_data[:5].cpu().numpy()
                    
                    center = (int(x), int(y))
                    size = (int(w), int(h))
                    rect = cv2.RotatedRect(center, size, float(rotation))
                    box_points = cv2.boxPoints(rect).astype(int)
                    
                    cv2.polylines(frame, [box_points], True, (0, 255, 0), 2)
        
        cv2.imshow('OBB Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser("Inspect trained YOLO model on video (frame-by-frame)")
    ap.add_argument("--model", type=str, default="fpv_gate_bb", 
                   help="Model key from registry (e.g., 'fpv_gate_bb', 'fpv_gate_obb') or direct path to .pt file")
    ap.add_argument("--video", type=str, default=None, help="Path to input .mp4/.mov etc")
    ap.add_argument("--list-models", action="store_true", help="List all available models and exit")
    ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    ap.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    ap.add_argument("--imgsz", type=int, default=640, help="Inference image size (ultralytics imgsz)")
    ap.add_argument("--maxdet", type=int, default=50, help="Max detections per frame")

    # Playback / stepping
    ap.add_argument("--start", type=int, default=0, help="Start at frame index")
    ap.add_argument("--step", type=int, default=1, help="How many frames to move on each step (default 1)")
    ap.add_argument("--play-fps", type=float, default=20.0, help="FPS while playing")

    args = ap.parse_args()
    
    # Handle --list-models
    if args.list_models:
        print("Available Models:")
        for key, desc in ModelRegistry.list_available_models().items():
            print(f"  {key}: {desc}")
        exit(0)
    
    # Check video is provided
    if not args.video:
        print("[ERROR] --video is required (or use --list-models)")
        exit(1)
    
    # Load model configuration
    model_config = ModelRegistry.get_model(args.model)
    if model_config is None:
        # Assume it's a direct path
        print(f"Model '{args.model}' not in registry, assuming direct path...")
        model_path = args.model
        model_config = None
    else:
        if not model_config.exists():
            print(f"[ERROR] Model file not found: {model_config.model_path}")
            exit(1)
        model_path = model_config.model_path

    model = YOLO(model_path)

    # If you want to force a mapping (in case your model names don’t match), edit DEFAULT_CLASS_NAMES above.
    class_names = get_class_names(model, override=None)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.video}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or -1

    # Seek to start frame
    if args.start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)

    playing = False
    frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) or 0
    last_render_time = time.time()

    print("\nControls:")
    print("  SPACE  : play/pause")
    print("  n      : next frame (or step)")
    print("  b      : back one step")
    print("  j      : jump +100 frames")
    print("  k      : jump -100 frames")
    print("  q / ESC: quit\n")

    while True:
        if playing:
            # throttle to play-fps
            now = time.time()
            if now - last_render_time < (1.0 / max(args.play_fps, 1e-6)):
                key = cv2.waitKey(1) & 0xFF
                if key in (ord(" "), ord("q"), 27, ord("n"), ord("b"), ord("j"), ord("k")):
                    # handle immediately
                    pass
                else:
                    continue
            last_render_time = now

        ok, frame = cap.read()
        if not ok:
            print("End of video.")
            break

        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1

        # Run model
        res = model.predict(
            frame,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            max_det=args.maxdet,
            verbose=False,
        )[0]

        # Draw detections - auto-detect model type from results
        if hasattr(res, 'obb') and res.obb is not None and len(res.obb) > 0:
            # OBB format: [x_center, y_center, width, height, rotation, conf, cls_id]
            for obb_box in res.obb.data:
                try:
                    x_center, y_center, width, height, rotation = obb_box[:5].cpu().numpy()
                    conf = float(obb_box[5])
                    cls_id = int(obb_box[6])
                except (IndexError, AttributeError):
                    # Fallback in case tensor layout differs
                    continue
                
                name = class_names.get(cls_id, f"class_{cls_id}")
                color = DEFAULT_COLORS.get(name, DEFAULT_COLORS["unknown"])
                
                # Draw rotated rectangle
                center = (int(x_center), int(y_center))
                size = (int(width), int(height))
                rect = cv2.RotatedRect(center, size, float(rotation))
                box_points = cv2.boxPoints(rect).astype(int)
                cv2.polylines(frame, [box_points], True, color, 3)
                
                label = f"{name} {conf:.2f} (id={cls_id})"
                cv2.putText(frame, label, (int(x_center), max(18, int(y_center) - 30)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        
        elif hasattr(res, 'keypoints') and res.keypoints is not None and len(res.keypoints) > 0:
            # Pose/Keypoint format - draw both boxes and keypoints
            for b in res.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                conf = float(b.conf[0])
                cls_id = int(b.cls[0])

                name = class_names.get(cls_id, f"class_{cls_id}")
                color = DEFAULT_COLORS.get(name, DEFAULT_COLORS["unknown"])

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                label = f"{name} {conf:.2f} (id={cls_id})"
                cv2.putText(frame, label, (x1, max(18, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
            
            # Draw keypoints if available
            for keypoint_set in res.keypoints.data:
                for kp in keypoint_set:
                    x, y, conf = kp
                    if conf > 0.5:  # Only draw if confidence is high
                        cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 255), -1)
        
        elif res.boxes is not None and len(res.boxes) > 0:
            # Standard boxes (Bounding Box model)
            for b in res.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                conf = float(b.conf[0])
                cls_id = int(b.cls[0])

                name = class_names.get(cls_id, f"class_{cls_id}")
                color = DEFAULT_COLORS.get(name, DEFAULT_COLORS["unknown"])

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                label = f"{name} {conf:.2f} (id={cls_id})"
                cv2.putText(frame, label, (x1, max(18, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        # Legend + HUD
        draw_legend(frame, class_names, DEFAULT_COLORS)
        h, w = frame.shape[:2]
        hud = f"frame {frame_idx}/{total if total>0 else '?'}  conf={args.conf:.2f}  play={'ON' if playing else 'OFF'}"
        cv2.putText(frame, hud, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 240), 2, cv2.LINE_AA)

        cv2.imshow("Gate Model Inspector", frame)

        key = cv2.waitKey(0 if not playing else 1) & 0xFF

        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            playing = not playing
        elif key == ord("n"):
            # advance by step: easiest is to set capture position
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx + args.step)
        elif key == ord("b"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx - args.step))
        elif key == ord("j"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(max(0, frame_idx + 100), max(0, total - 1)))
        elif key == ord("k"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx - 100))
        # otherwise: continue (especially while playing)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
