from ultralytics import YOLO
import cv2
import pytesseract
import numpy as np
from PIL import Image

# Set Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class VisionProcessor:
    def __init__(self):
        print("Initializing YOLO model...")
        # Initialize YOLO model with reduced confidence threshold
        self.model = YOLO('yolov8n.pt')  # using the nano model for speed
        print("YOLO model loaded successfully")
        
    def detect_objects(self, frame):
        """Detect objects in the frame using YOLOv8."""
        results = self.model(frame)
        detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].numpy()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                
                if confidence > 0.5:  # confidence threshold
                    detections.append({
                        'class': class_name,
                        'confidence': confidence,
                        'box': [int(x1), int(y1), int(x2), int(y2)]
                    })
        
        return detections

    def extract_text(self, frame):
        """Extract text from the frame using Tesseract OCR."""
        # Convert frame to PIL Image
        if isinstance(frame, np.ndarray):
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
              # Extract text
        text = pytesseract.image_to_string(frame)
        return text.strip()
        
    def process_frame(self, frame):
        """Process a frame and return both object detections and text."""
        print("Starting frame processing...")
        print("Detecting objects...")
        objects = self.detect_objects(frame)
        print(f"Found {len(objects)} objects")
        
        print("Extracting text...")
        text = self.extract_text(frame)
        print("Text extraction complete")
        
        return {
            'objects': objects,
            'text': text
        }

    def create_scene_description(self, frame):
        """Create a natural language description of the scene."""
        results = self.process_frame(frame)
        
        # Create description of detected objects
        object_desc = []
        if results['objects']:
            object_counts = {}
            for obj in results['objects']:
                class_name = obj['class']
                object_counts[class_name] = object_counts.get(class_name, 0) + 1
                
            for class_name, count in object_counts.items():
                if count > 1:
                    object_desc.append(f"{count} {class_name}s")
                else:
                    object_desc.append(f"a {class_name}")
                    
        # Add any text found in the image
        text_desc = ""
        if results['text']:
            text_desc = f" Text found in the image reads: {results['text']}"
            
        # Combine descriptions
        if object_desc:
            objects_text = ", ".join(object_desc[:-1])
            if len(object_desc) > 1:
                objects_text += f" and {object_desc[-1]}"
            else:
                objects_text = object_desc[0]
            scene_desc = f"I can see {objects_text}.{text_desc}"
        else:
            scene_desc = "I don't see any recognizable objects in this image."
            if text_desc:
                scene_desc += text_desc
                
        return scene_desc
