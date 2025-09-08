import random
import cv2 as cv
import numpy as np
from ultralytics import YOLO
import time
import os

class ObjectDetector:
    """Enhanced object detection class with improved performance and features."""
    
    def __init__(self, model_path="weights/yolov8n.pt", class_file="models/utils/coco.txt"):
        """Initialize the object detector.
        
        Args:
            model_path (str): Path to YOLO model weights
            class_file (str): Path to class names file
        """
        self.model_path = model_path
        self.class_file = class_file
        
        # Load class names
        self.class_list = self._load_class_names()
        
        # Generate consistent colors for each class
        self.detection_colors = self._generate_colors()
        
        # Load YOLO model
        self.model = self._load_model()
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # Detection parameters - more balanced
        self.confidence_threshold = 0.4  # Higher threshold for better accuracy
        self.nms_threshold = 0.5
        
    def _load_class_names(self):
        """Load class names from file."""
        try:
            with open(self.class_file, "r") as f:
                return f.read().strip().split("\n")
        except FileNotFoundError:
            print(f"Warning: Class file {self.class_file} not found. Using default classes.")
            return [f"class_{i}" for i in range(80)]
    
    def _generate_colors(self):
        """Generate consistent colors for each class."""
        colors = []
        for i in range(len(self.class_list)):
            # Use a deterministic approach for consistent colors
            random.seed(i)
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            colors.append((b, g, r))
        return colors
    
    def _load_model(self):
        """Load YOLO model with error handling."""
        try:
            return YOLO(self.model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    def detect_objects(self, frame):
        """Detect objects in a frame.
        
        Args:
            frame (np.array): Input frame
            
        Returns:
            tuple: (processed_frame, detections_info)
        """
        if self.model is None:
            return frame, []
        
        # Run detection with balanced settings
        results = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            save=False,
            verbose=False,
            imgsz=320,  # Slightly larger image size for better accuracy
            device='cpu',
            half=False,
            augment=False,
            agnostic_nms=False, # Better NMS
            max_det=10,  # More detections
            iou=0.5,  # Standard IoU threshold
            classes=[0, 1, 2, 3, 5, 7, 9],  # Added traffic light
            retina_masks=False,
            show=False,
            save_txt=False,
            save_conf=False,
            save_crop=False,
            show_labels=True,
            show_conf=True,
            line_width=2 # Thicker lines
        )
        
        # Process results
        processed_frame = frame.copy()
        detections_info = []
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            
            for i in range(len(boxes)):
                box = boxes[i]
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                bbox = box.xyxy[0].cpu().numpy()
                
                # Draw bounding box
                x1, y1, x2, y2 = map(int, bbox)
                color = self.detection_colors[cls_id % len(self.detection_colors)]
                
                cv.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label with confidence
                label = f"{self.class_list[cls_id]}: {confidence:.2f}"
                label_size = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                # Draw label background
                cv.rectangle(processed_frame, (x1, y1 - label_size[1] - 10), 
                           (x1 + label_size[0], y1), color, -1)
                
                # Draw label text
                cv.putText(processed_frame, label, (x1, y1 - 5),
                          cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                detections_info.append({
                    'class': self.class_list[cls_id],
                    'confidence': confidence,
                    'bbox': bbox
                })
        
        return processed_frame, detections_info
    
    def update_fps(self):
        """Update FPS counter."""
        self.fps_counter += 1
        if self.fps_counter % 30 == 0:  # Update every 30 frames
            current_time = time.time()
            self.current_fps = 30 / (current_time - self.fps_start_time)
            self.fps_start_time = current_time
    
    def draw_fps(self, frame):
        """Draw FPS counter on frame."""
        fps_text = f"FPS: {self.current_fps:.1f}"
        cv.putText(frame, fps_text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return frame

def main():
    """Main function to run object detection."""
    # Initialize detector
    detector = ObjectDetector()
    
    # Initialize camera
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return
    
    # Set camera properties for better performance
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv.CAP_PROP_FPS, 30)
    
    print("Object Detection started. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame. Exiting...")
            break
        
        # Detect objects
        processed_frame, detections = detector.detect_objects(frame)
        
        # Update FPS and draw it
        detector.update_fps()
        processed_frame = detector.draw_fps(processed_frame)
        
        # Draw detection count
        count_text = f"Objects: {len(detections)}"
        cv.putText(processed_frame, count_text, (10, 70), 
                  cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display frame
        cv.imshow("Enhanced Object Detector", processed_frame)
        
        # Check for quit
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()