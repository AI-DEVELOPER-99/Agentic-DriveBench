"""Agent 1: Perception Agent - Object detection and depth estimation."""
from typing import List, Dict, Any, Optional
from .ollama_client import OllamaClient
from ultralytics import YOLO
import cv2
import numpy as np

# yolov8n.pt

class PerceptionAgent:
    """Uses YOLO for fast object detection with bounding boxes and VLM for scene understanding."""
    
    def __init__(self, client: OllamaClient, vlm_model: str = "llava:latest", yolo_model: str = "yolo26s.pt"):
        self.client = client
        self.vlm_model = vlm_model
        # Initialize YOLO for fast object detection
        self.yolo = YOLO(yolo_model)
        # Classes relevant for autonomous driving
        self.driving_classes = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus',
            7: 'truck', 9: 'traffic light', 11: 'stop sign', 12: 'parking meter',
            13: 'bench'
        }
    
    def perceive(self, images: List[str], question: str = "", use_vlm: bool = False) -> Dict[str, Any]:
        """Detect objects and their attributes from images using YOLO + optional VLM.
        
        Args:
            images: List of camera image paths
            question: Optional question to focus perception
            use_vlm: Whether to use VLM for additional scene understanding
            
        Returns:
            Detection results with objects, bounding boxes, positions, and attributes
        """
        # Run YOLO detection on all images
        all_detections = []
        camera_views = ['front', 'front_left', 'front_right', 'back', 'back_left', 'back_right']
        
        for idx, img_path in enumerate(images):
            view = camera_views[idx] if idx < len(camera_views) else f'camera_{idx}'
            detections = self._detect_with_yolo(img_path, view)
            all_detections.extend(detections)
        
        # Generate structured description from YOLO detections
        description = self._generate_description_from_detections(all_detections)
        
        result = {
            "detections": all_detections,
            "description": description,
            "detection_method": "yolo"
        }
        
        # Optionally use VLM for additional context
        if use_vlm:
            vlm_response = self._get_vlm_context(images, question, all_detections)
            result["vlm_context"] = vlm_response
        
        return result
    
    def _detect_with_yolo(self, image_path: str, camera_view: str) -> List[Dict[str, Any]]:
        """Run YOLO detection on a single image.
        
        Args:
            image_path: Path to image
            camera_view: Camera view name (front, back, etc.)
            
        Returns:
            List of detected objects with bounding boxes
        """
        results = self.yolo(image_path, verbose=False)[0]
        detections = []
        
        img = cv2.imread(image_path)
        img_height, img_width = img.shape[:2]
        
        for box in results.boxes:
            cls_id = int(box.cls[0])
            # Filter for driving-relevant objects
            if cls_id in self.driving_classes or cls_id in range(0, 80):
                conf = float(box.conf[0])
                if conf > 0.3:  # Confidence threshold
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Calculate center position
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    # Determine position relative to ego vehicle
                    position = self._get_relative_position(center_x, center_y, img_width, img_height, camera_view)
                    distance = self._estimate_distance(y2, img_height)
                    
                    detection = {
                        "class": self.yolo.names[cls_id],
                        "confidence": round(conf, 3),
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "center": [int(center_x), int(center_y)],
                        "camera_view": camera_view,
                        "position": position,
                        "distance": distance
                    }
                    detections.append(detection)
        
        return detections
    
    def _get_relative_position(self, x: float, y: float, width: int, height: int, view: str) -> str:
        """Determine object position relative to ego vehicle."""
        # Horizontal position in image
        if x < width * 0.33:
            h_pos = "left"
        elif x < width * 0.66:
            h_pos = "center"
        else:
            h_pos = "right"
        
        # Combine with camera view
        if view == "front":
            return f"front-{h_pos}" if h_pos != "center" else "front-center"
        elif view == "back":
            return f"back-{h_pos}" if h_pos != "center" else "back-center"
        else:
            return f"{view}-{h_pos}"
    
    def _estimate_distance(self, bottom_y: float, img_height: int) -> str:
        """Estimate distance category based on vertical position."""
        ratio = bottom_y / img_height
        if ratio > 0.8:
            return "close"
        elif ratio > 0.5:
            return "medium"
        else:
            return "far"
    
    def _generate_description_from_detections(self, detections: List[Dict[str, Any]]) -> str:
        """Generate natural language description from YOLO detections."""
        if not detections:
            return "No objects detected in the scene."
        
        # Categorize detections
        vehicles = []
        pedestrians = []
        traffic_elements = []
        
        for det in detections:
            cls = det["class"]
            if cls in ["car", "truck", "bus", "motorcycle"]:
                vehicles.append(det)
            elif cls == "person":
                pedestrians.append(det)
            elif cls in ["traffic light", "stop sign"]:
                traffic_elements.append(det)
        
        description_parts = []
        
        # Describe vehicles
        if vehicles:
            vehicle_desc = f"Detected {len(vehicles)} vehicle(s): "
            vehicle_details = []
            for v in vehicles[:5]:  # Limit to top 5
                vehicle_details.append(f"{v['class']} at {v['position']} ({v['distance']} distance)")
            description_parts.append(vehicle_desc + ", ".join(vehicle_details))
        
        # Describe pedestrians
        if pedestrians:
            ped_desc = f"Detected {len(pedestrians)} pedestrian(s): "
            ped_details = []
            for p in pedestrians[:3]:
                ped_details.append(f"at {p['position']} ({p['distance']} distance)")
            description_parts.append(ped_desc + ", ".join(ped_details))
        
        # Describe traffic elements
        if traffic_elements:
            traffic_desc = f"Traffic elements: "
            traffic_details = [f"{t['class']} at {t['position']}" for t in traffic_elements]
            description_parts.append(traffic_desc + ", ".join(traffic_details))
        
        return ". ".join(description_parts) + "."
    
    def _get_vlm_context(self, images: List[str], question: str, detections: List[Dict]) -> str:
        """Use VLM for additional scene understanding."""
        prompt = f"""You are analyzing a driving scene. YOLO has detected these objects:
{self._format_detections_for_vlm(detections)}

Provide additional context about:
- Road conditions and obstacles
- Movement patterns and behavior of detected objects
- Any safety concerns or notable scene characteristics

{f'Focus on: {question}' if question else ''}

Provide brief additional insights:"""
        
        response = self.client.chat_vlm(
            model=self.vlm_model,
            prompt=prompt,
            images=images,
            temperature=0.0
        )
        
        return response
    
    def _format_detections_for_vlm(self, detections: List[Dict]) -> str:
        """Format YOLO detections for VLM prompt."""
        if not detections:
            return "No objects detected"
        
        summary = []
        for det in detections[:10]:  # Top 10
            summary.append(f"- {det['class']} at {det['position']} ({det['distance']})")
        return "\n".join(summary)
    
    def _parse_detection(self, response: str) -> List[Dict[str, str]]:
        """Parse VLM detection response into structured objects."""
        objects = []
        lines = response.strip().split('\n')
        
        for line in lines:
            if line.strip().startswith('-'):
                obj = {}
                parts = line.strip('- ').split(',')
                for part in parts:
                    if ':' in part:
                        key, value = part.split(':', 1)
                        obj[key.strip().lower()] = value.strip()
                if obj:
                    objects.append(obj)
        
        return objects
