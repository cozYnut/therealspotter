"""
Model configuration system for supporting multiple YOLO model types.
Centralizes model definitions and provides utilities for working with different detection modes.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, List
from pathlib import Path


class ModelType(Enum):
    """Enum for supported YOLO model types"""
    BB = "bb"  # Bounding Box (standard YOLO detection)
    OBB = "obb"  # Oriented Bounding Box (rotated detection)
    KEYPOINTS = "keypoints"  # Key-Points (pose/landmark detection)


@dataclass
class ModelConfig:
    """Configuration for a YOLO model"""
    model_type: ModelType
    name: str
    description: str
    model_path: str
    task: str  # "detect", "obb", "pose", etc.
    has_rotation: bool  # Whether output includes rotation angle
    has_keypoints: bool  # Whether output includes keypoints
    num_keypoints: Optional[int] = None  # Number of keypoints (if applicable)
    default_conf: float = 0.25  # Default confidence threshold
    default_maxdet: int = 50  # Default max detections per frame
    
    def exists(self) -> bool:
        """Check if model file exists"""
        return Path(self.model_path).exists()
    
    def validate(self) -> bool:
        """Validate model configuration"""
        if not self.exists():
            print(f"Warning: Model file not found: {self.model_path}")
            return False
        
        # Validate consistency between model type and task
        if self.model_type == ModelType.BB and self.task not in ["detect"]:
            print(f"Warning: BB model should have task='detect', got '{self.task}'")
            return False
        
        if self.model_type == ModelType.OBB and self.task not in ["obb"]:
            print(f"Warning: OBB model should have task='obb', got '{self.task}'")
            return False
        
        if self.model_type == ModelType.KEYPOINTS and self.task not in ["pose"]:
            print(f"Warning: KEYPOINTS model should have task='pose', got '{self.task}'")
            return False
        
        return True


class ModelRegistry:
    """Registry of all available model configurations"""
    
    # Define all available models here
    MODELS: Dict[str, ModelConfig] = {
        # Bounding Box Models (standard YOLO)
        "fpv_gate_bb": ModelConfig(
            model_type=ModelType.BB,
            name="FPV Gate BB",
            description="FPV gate detection with standard bounding boxes",
            model_path="current_best_non_vocab.pt",
            task="detect",
            has_rotation=False,
            has_keypoints=False,
            default_conf=0.25,
            default_maxdet=50,
        ),
        "yolov8n": ModelConfig(
            model_type=ModelType.BB,
            name="YOLOv8 Nano",
            description="YOLOv8 Nano pretrained model for general detection",
            model_path="yolov8n.pt",
            task="detect",
            has_rotation=False,
            has_keypoints=False,
            default_conf=0.25,
            default_maxdet=50,
        ),
        "yolov8m": ModelConfig(
            model_type=ModelType.BB,
            name="YOLOv8 Medium",
            description="YOLOv8 Medium pretrained model for general detection",
            model_path="yolov8m.pt",
            task="detect",
            has_rotation=False,
            has_keypoints=False,
            default_conf=0.25,
            default_maxdet=50,
        ),
        
        # Oriented Bounding Box Models
        "fpv_gate_obb": ModelConfig(
            model_type=ModelType.OBB,
            name="FPV Gate OBB",
            description="FPV gate detection with oriented bounding boxes",
            model_path="runs/obb/fpv_gate_obb_train/weights/best.pt",
            task="obb",
            has_rotation=True,
            has_keypoints=False,
            default_conf=0.25,
            default_maxdet=50,
        ),
        "yolov8m-obb": ModelConfig(
            model_type=ModelType.OBB,
            name="YOLOv8 Medium OBB",
            description="YOLOv8 Medium pretrained OBB model",
            model_path="yolov8m-obb.pt",
            task="obb",
            has_rotation=True,
            has_keypoints=False,
            default_conf=0.25,
            default_maxdet=50,
        ),
        
        # Key-Points Models
        "yolov8n-pose": ModelConfig(
            model_type=ModelType.KEYPOINTS,
            name="YOLOv8 Nano Pose",
            description="YOLOv8 Nano for pose/keypoint detection (17 keypoints)",
            model_path="yolov8n-pose.pt",
            task="pose",
            has_rotation=False,
            has_keypoints=True,
            num_keypoints=17,
            default_conf=0.25,
            default_maxdet=50,
        ),
        "yolov8m-pose": ModelConfig(
            model_type=ModelType.KEYPOINTS,
            name="YOLOv8 Medium Pose",
            description="YOLOv8 Medium for pose/keypoint detection (17 keypoints)",
            model_path="yolov8m-pose.pt",
            task="pose",
            has_rotation=False,
            has_keypoints=True,
            num_keypoints=17,
            default_conf=0.25,
            default_maxdet=50,
        ),
    }
    
    @classmethod
    def get_model(cls, model_key: str) -> Optional[ModelConfig]:
        """Get model config by key"""
        return cls.MODELS.get(model_key)
    
    @classmethod
    def get_models_by_type(cls, model_type: ModelType) -> Dict[str, ModelConfig]:
        """Get all models of a specific type"""
        return {k: v for k, v in cls.MODELS.items() if v.model_type == model_type}
    
    @classmethod
    def list_models(cls) -> Dict[str, str]:
        """List all available models with descriptions"""
        return {k: f"{v.name} - {v.description}" for k, v in cls.MODELS.items()}
    
    @classmethod
    def list_available_models(cls) -> Dict[str, str]:
        """List only available (existing) models"""
        return {
            k: f"{v.name} - {v.description}"
            for k, v in cls.MODELS.items()
            if v.exists()
        }
    
    @classmethod
    def add_model(cls, key: str, config: ModelConfig) -> None:
        """Add or override a model configuration"""
        cls.MODELS[key] = config
    
    @classmethod
    def validate_all(cls) -> bool:
        """Validate all model configurations"""
        all_valid = True
        for key, config in cls.MODELS.items():
            if not config.validate():
                all_valid = False
        return all_valid


# Utility functions for working with detection results

def detection_to_yolo_line(
    bbox: tuple,
    img_w: int,
    img_h: int,
    class_id: int,
    model_config: ModelConfig,
    rotation_angle: float = 0.0,
    keypoints: Optional[List[float]] = None,
) -> str:
    """
    Convert detection to YOLO format label line.
    
    Args:
        bbox: (x1, y1, x2, y2) in pixel coordinates
        img_w, img_h: Image dimensions
        class_id: Class index
        model_config: ModelConfig object
        rotation_angle: Rotation angle in degrees (for OBB models)
        keypoints: Keypoint data (for pose models)
    
    Returns:
        YOLO format label line
    """
    x1, y1, x2, y2 = bbox
    x_center = ((x1 + x2) / 2.0) / float(img_w)
    y_center = ((y1 + y2) / 2.0) / float(img_h)
    width = (x2 - x1) / float(img_w)
    height = (y2 - y1) / float(img_h)
    
    # Build label based on model type
    parts = [str(int(class_id)), f"{x_center:.6f}", f"{y_center:.6f}", 
             f"{width:.6f}", f"{height:.6f}"]
    
    if model_config.has_rotation:
        # Normalize rotation angle to [0, 360)
        angle = float(rotation_angle) % 360.0
        parts.append(f"{angle:.6f}")
    
    if model_config.has_keypoints and keypoints is not None:
        # Add keypoint coordinates and confidence scores
        parts.extend([f"{kp:.6f}" for kp in keypoints])
    
    return " ".join(parts)


def get_label_columns_count(model_config: ModelConfig) -> int:
    """Get number of columns in output label format"""
    count = 5  # class_id + 4 bbox values
    
    if model_config.has_rotation:
        count += 1  # rotation angle
    
    if model_config.has_keypoints and model_config.num_keypoints:
        count += model_config.num_keypoints * 3  # x, y, confidence per keypoint
    
    return count


if __name__ == "__main__":
    # Quick test
    print("Available Models:")
    for key, desc in ModelRegistry.list_available_models().items():
        print(f"  {key}: {desc}")
    
    print("\nAll Registered Models:")
    for key, desc in ModelRegistry.list_models().items():
        model = ModelRegistry.get_model(key)
        print(f"  {key}: {desc} (exists: {model.exists()})")
