import os
import cv2
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional
import math
import numpy as np
import time  # ADD THIS

# ----------------------------
# Helpers
# ----------------------------

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def yolo_from_xyxy(x1, y1, x2, y2, w, h):
    # Ensure proper ordering
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    xc = x1 + bw / 2.0
    yc = y1 + bh / 2.0
    return (xc / w, yc / h, bw / w, bh / h)

def xyxy_from_two_points(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))

def nice_frame_name(video_stem: str, frame_idx: int) -> str:
    return f"{video_stem}_{frame_idx:06d}"

# ----------------------------
# UI state
# ----------------------------

@dataclass
class Box:
    cls_id: int
    xyxy: Tuple[int, int, int, int]
    rotation: float = 0.0

class Annotator:
    def __init__(self, class_names: List[str], use_obb: bool = False):
        self.class_names = class_names
        self.cur_cls = 0
        self.use_obb = use_obb

        self.boxes: List[Box] = []
        self.drawing = False
        self.p1 = (0, 0)
        self.p2 = (0, 0)

        # NEW: 3-click OBB mode state
        self.obb_click_count = 0
        self.obb_points = []  # [top_right, top_left, height_point]
        self.obb_points_preview = None

        self.flash_msg = ""
        self.flash_until = 0.0

    def set_flash(self, msg: str, now: float, seconds: float = 1.2):
        self.flash_msg = msg
        self.flash_until = now + seconds

    def compute_obb_from_three_points(self, p_tr, p_tl, p_height):
        """
        Compute OBB from 3 points:
        - p_tr: top-right corner
        - p_tl: top-left corner
        - p_height: point determining height
        
        Returns: (cx, cy, width, height, rotation_angle_degrees)
        """
        x_tr, y_tr = float(p_tr[0]), float(p_tr[1])
        x_tl, y_tl = float(p_tl[0]), float(p_tl[1])
        x_h, y_h = float(p_height[0]), float(p_height[1])

        # Top edge vector (from top-left to top-right)
        top_vec = np.array([x_tr - x_tl, y_tr - y_tl], dtype=np.float32)
        top_len = np.linalg.norm(top_vec)
        if top_len < 1:
            return None  # degenerate
        
        top_unit = top_vec / top_len
        width = top_len

        # Height vector (from top-left to height point)
        height_vec = np.array([x_h - x_tl, y_h - y_tl], dtype=np.float32)
        # Perpendicular unit vector (90° clockwise instead of counterclockwise)
        perp_unit = np.array([top_unit[1], -top_unit[0]])  # FIX: Changed from [-top_unit[1], top_unit[0]]
        # Project height_vec onto perpendicular
        height = abs(np.dot(height_vec, perp_unit))
        if height < 1:
            height = 1

        # Rotation angle in degrees
        rotation = math.degrees(math.atan2(top_vec[1], top_vec[0]))

        # Center of box
        center_x = x_tl + top_unit[0] * width / 2.0 + perp_unit[0] * height / 2.0
        center_y = y_tl + top_unit[1] * width / 2.0 + perp_unit[1] * height / 2.0

        return (center_x, center_y, width, height, rotation)

    def on_mouse(self, event, x, y, flags, param):
        if not self.use_obb:
            # Standard 2-click bbox mode
            if event == cv2.EVENT_LBUTTONDOWN:
                self.drawing = True
                self.p1 = (x, y)
                self.p2 = (x, y)
            elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
                self.p2 = (x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                self.drawing = False
                self.p2 = (x, y)
                x1, y1, x2, y2 = xyxy_from_two_points(self.p1, self.p2)
                # ignore tiny boxes
                if (x2 - x1) >= 4 and (y2 - y1) >= 4:
                    self.boxes.append(Box(
                        cls_id=self.cur_cls, 
                        xyxy=(x1, y1, x2, y2),
                        rotation=0.0
                    ))
        else:
            # NEW: 3-click OBB mode
            if event == cv2.EVENT_LBUTTONDOWN:
                self.obb_points.append((x, y))
                self.obb_click_count += 1

                if self.obb_click_count == 3:
                    # All 3 points collected - compute OBB
                    result = self.compute_obb_from_three_points(
                        self.obb_points[0],  # top-right
                        self.obb_points[1],  # top-left
                        self.obb_points[2]   # height point
                    )
                    if result:
                        cx, cy, w, h, rot = result
                        # Convert to xyxy format
                        x1 = int(cx - w / 2)
                        y1 = int(cy - h / 2)
                        x2 = int(cx + w / 2)
                        y2 = int(cy + h / 2)
                        
                        # Ensure min size
                        if (x2 - x1) >= 4 and (y2 - y1) >= 4:
                            self.boxes.append(Box(
                                cls_id=self.cur_cls,
                                xyxy=(x1, y1, x2, y2),
                                rotation=rot
                            ))
                            self.set_flash(f"OBB created (rotation: {rot:.1f}°)", time.time(), 1.0)  # FIX: time is now imported
                    
                    # Reset for next box
                    self.obb_click_count = 0
                    self.obb_points = []
                    self.obb_points_preview = None

            elif event == cv2.EVENT_MOUSEMOVE and self.obb_click_count > 0:
                # Preview next point as you move mouse
                self.obb_points_preview = (x, y)

    def undo(self):
        if self.boxes:
            self.boxes.pop()
        # Also reset OBB drawing if in progress
        if self.use_obb and self.obb_click_count > 0:
            self.obb_click_count = 0
            self.obb_points = []
            self.obb_points_preview = None

    def clear(self):
        self.boxes = []
        self.obb_click_count = 0
        self.obb_points = []
        self.obb_points_preview = None

    def rotate_current_box(self, delta: float, now: float):
        """Rotate the last drawn box"""
        if not self.boxes:
            return
        self.boxes[-1].rotation = (self.boxes[-1].rotation + delta) % 360.0
        self.set_flash(f"Rotation: {self.boxes[-1].rotation:.1f}°", now, 0.5)

    def draw_overlay(self, img, now: float):
        H, W = img.shape[:2]

        # draw existing boxes
        for b in self.boxes:
            x1, y1, x2, y2 = b.xyxy
            
            if self.use_obb:
                # Draw rotated rectangle for OBB
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                w = abs(x2 - x1)
                h = abs(y2 - y1)
                
                rect = cv2.RotatedRect((cx, cy), (w, h), b.rotation)
                box_points = cv2.boxPoints(rect)
                box_points = np.int32(box_points)
                cv2.polylines(img, [box_points], True, (0, 255, 0), 2)
                
                # Draw center + rotation indicator line
                cv2.circle(img, (int(cx), int(cy)), 3, (0, 255, 0), -1)
                angle_rad = math.radians(b.rotation)
                indicator_len = max(w, h) / 3
                end_x = cx + indicator_len * math.cos(angle_rad)
                end_y = cy + indicator_len * math.sin(angle_rad)
                cv2.line(img, (int(cx), int(cy)), (int(end_x), int(end_y)), (0, 255, 0), 2)
            else:
                # Standard axis-aligned box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            name = self.class_names[b.cls_id] if 0 <= b.cls_id < len(self.class_names) else str(b.cls_id)
            rot_text = f" {b.rotation:.0f}°" if (self.use_obb and b.rotation != 0.0) else ""
            cv2.putText(img, name + rot_text, (int(x1), max(15, int(y1) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 2, cv2.LINE_AA)

        # NEW: Draw in-progress 3-click OBB
        if self.use_obb and self.obb_click_count > 0:
            # Draw collected points as colored circles
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Red, Green, Blue
            labels = ["P1 (top-right)", "P2 (top-left)", "P3 (height)"]
            
            # FIX: Use min() to prevent index out of range
            for i in range(min(len(self.obb_points), len(colors))):
                pt = self.obb_points[i]
                cv2.circle(img, pt, 6, colors[i], -1)
                cv2.circle(img, pt, 6, (255, 255, 255), 1)
                cv2.putText(img, labels[i], (pt[0] + 10, pt[1] - 8), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2)

            # Draw edges between collected points
            if self.obb_click_count >= 2:
                cv2.line(img, self.obb_points[0], self.obb_points[1], (150, 150, 255), 2, cv2.LINE_AA)

            # Draw preview line to mouse cursor
            if self.obb_points_preview and self.obb_click_count < 3:
                preview_color = colors[self.obb_click_count]
                cv2.line(img, self.obb_points[-1], self.obb_points_preview, preview_color, 1, cv2.LINE_AA)
                cv2.circle(img, self.obb_points_preview, 4, preview_color, 1)

        # HUD
        if self.use_obb:
            mode_str = "OBB mode (3-click)"
            click_status = f" | Clicks: {self.obb_click_count}/3" if self.obb_click_count > 0 else ""
        else:
            mode_str = "BBOX mode (2-click)"
            click_status = ""
        
        hud1 = f"[{mode_str}] Class: [{self.cur_cls}] {self.class_names[self.cur_cls]}   Boxes: {len(self.boxes)}{click_status}"
        
        if self.use_obb:
            hud2 = "Keys: [a/d] prev/next  [j/k] -/+skip  [1..9] class  [</>] rotate±5°  [u] undo  [c] clear  [s] save  [q] quit"
        else:
            hud2 = "Keys: [a/d] prev/next  [j/k] -/+skip  [1..9] class  [u] undo  [c] clear  [s] save  [q] quit"
        
        cv2.putText(img, hud1, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 240), 2, cv2.LINE_AA)
        cv2.putText(img, hud2, (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 2, cv2.LINE_AA)

        if self.use_obb and self.obb_click_count > 0:
            instructions = ["Click top-right corner", "Click top-left corner", "Click height reference point"]
            cv2.putText(img, f"→ {instructions[self.obb_click_count - 1]}", (10, 86), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

        if now < self.flash_until and self.flash_msg:
            cv2.putText(img, self.flash_msg, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)

# ----------------------------
# Main
# ----------------------------

def save_current_frame(frame, frame_idx: int, force_empty: bool, now: float, ann: Annotator, **kwargs):
    name = nice_frame_name(kwargs['video_stem'], frame_idx)
    out_img = os.path.join(kwargs['img_dir'], f"{name}.jpg")
    out_lab = os.path.join(kwargs['lab_dir'], f"{name}.txt")

    Hf, Wf = frame.shape[:2]

    # Build label lines
    lines = []
    if not force_empty:
        for b in ann.boxes:
            x1, y1, x2, y2 = b.xyxy
            x1 = clamp(x1, 0, Wf - 1)
            x2 = clamp(x2, 0, Wf - 1)
            y1 = clamp(y1, 0, Hf - 1)
            y2 = clamp(y2, 0, Hf - 1)
            
            if ann.use_obb:
                # OBB format: cls_id cx cy width height rotation
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                bw = x2 - x1
                bh = y2 - y1
                lines.append(f"{b.cls_id} {cx/Wf:.6f} {cy/Hf:.6f} {bw/Wf:.6f} {bh/Hf:.6f} {b.rotation:.1f}")
            else:
                # Standard YOLO bbox format
                xc, yc, bw, bh = yolo_from_xyxy(x1, y1, x2, y2, Wf, Hf)
                lines.append(f"{b.cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

    # Write label file
    with open(out_lab, "w") as f:
        if lines:
            f.write("\n".join(lines) + "\n")
        else:
            f.write("")

    # Write image
    cv2.imwrite(out_img, frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(kwargs['jpg_quality'])])

    if force_empty or (len(lines) == 0):
        ann.set_flash(f"SAVED EMPTY: {os.path.basename(out_img)}", now, 1.3)
    else:
        ann.set_flash(f"SAVED: {os.path.basename(out_img)} (+labels)", now, 1.3)

def main():
    ap = argparse.ArgumentParser(description="Video -> YOLO annotator (3-click OBB or 2-click BBOX)")
    ap.add_argument("--video", required=True, help="Path to .mp4")
    ap.add_argument("--out", required=True, help="Output dataset root")
    ap.add_argument("--split", default="train", choices=["train", "val", "test"], help="Which split to write into")
    ap.add_argument("--classes", default="square,circle,arch,flagpole", help="Comma-separated class names")
    ap.add_argument("--obb", action="store_true", help="Enable OBB (3-click) mode")
    ap.add_argument("--start", type=int, default=0, help="Start frame index")
    ap.add_argument("--step", type=int, default=1, help="Frame step when moving next/prev")
    ap.add_argument("--resize", type=str, default=None, help="Optional resize WxH")
    ap.add_argument("--jpg-quality", type=int, default=95, help="JPEG quality 0-100")
    args = ap.parse_args()

    class_names = [c.strip() for c in args.classes.split(",") if c.strip()]
    if not class_names:
        raise SystemExit("No classes provided.")

    # Prepare output dirs
    img_dir = os.path.join(args.out, "images", args.split)
    lab_dir = os.path.join(args.out, "labels", args.split)
    ensure_dir(img_dir)
    ensure_dir(lab_dir)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video: {args.video}")

    video_stem = os.path.splitext(os.path.basename(args.video))[0]
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else -1

    # Resize parsing
    resize_wh = None
    if args.resize:
        try:
            w_str, h_str = args.resize.lower().split("x")
            resize_wh = (int(w_str), int(h_str))
        except Exception:
            raise SystemExit("Bad --resize. Use like 1280x720")

    ann = Annotator(class_names=class_names, use_obb=args.obb)

    win = "YOLO Video Labeler"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, ann.on_mouse)

    def read_frame(idx: int):
        idx = max(0, idx)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            return None
        if resize_wh:
            frame = cv2.resize(frame, resize_wh, interpolation=cv2.INTER_AREA)
        return frame

    frame_idx = args.start
    frame = read_frame(frame_idx)
    if frame is None:
        raise SystemExit(f"Could not read frame {frame_idx} from video.")

    skip = max(1, args.step)

    while True:
        now = time.time()
        vis = frame.copy()
        ann.draw_overlay(vis, now)

        # bottom-left frame counter
        H, W = vis.shape[:2]
        count_text = f"Frame {frame_idx}" + (f"/{total_frames-1}" if total_frames > 0 else "")
        cv2.putText(vis, count_text, (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 240), 2, cv2.LINE_AA)

        cv2.imshow(win, vis)
        key = cv2.waitKey(20) & 0xFF

        if key == ord("q") or key == 27:
            break

        # class selection 1..9
        if ord("1") <= key <= ord("9"):
            cls = (key - ord("1"))
            if cls < len(class_names):
                ann.cur_cls = cls

        if key == ord("u"):
            ann.undo()

        if key == ord("c"):
            ann.clear()

        # OBB rotation controls
        if args.obb:
            if key == ord(">") or key == ord("."):
                ann.rotate_current_box(5.0, now)
            if key == ord("<") or key == ord(","):
                ann.rotate_current_box(-5.0, now)

        # prev/next
        if key == ord("d"):
            frame_idx += skip
            ann.clear()
            frame = read_frame(frame_idx)
            if frame is None:
                break

        if key == ord("a"):
            frame_idx = max(0, frame_idx - skip)
            ann.clear()
            frame = read_frame(frame_idx)
            if frame is None:
                break

        # adjust skip
        if key == ord("k"):
            skip = min(500, skip + 1)
            ann.set_flash(f"skip = {skip}", now, 0.8)

        if key == ord("j"):
            skip = max(1, skip - 1)
            ann.set_flash(f"skip = {skip}", now, 0.8)

        # save
        if key == ord("s"):
            save_current_frame(frame, frame_idx, force_empty=False, now=now, ann=ann,
                             video_stem=video_stem, img_dir=img_dir, lab_dir=lab_dir, 
                             jpg_quality=args.jpg_quality)
            frame_idx += skip
            ann.clear()
            frame = read_frame(frame_idx)
            if frame is None:
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")

if __name__ == "__main__":
    main()
