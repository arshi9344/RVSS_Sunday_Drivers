#!/usr/bin/env python
import sys
import os
import cv2
from ultralytics import YOLO
import torch

class DetectionInterface:
    def __init__(self, model_path, conf_threshold=0.95,area=1500):
        """
        Initialize the detection interface.
        
        Args:
            model_path (str): Path to the YOLO model weights
            conf_threshold (float): Confidence threshold for detections (0-1)
        """
        self.conf_threshold = conf_threshold
        self.area_threshold = area
        
        # Initialize YOLO model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        self.model = YOLO(model_path)
        self.model.to(self.device)
        
        # Suppress YOLO printing
        from ultralytics.utils import LOGGER
        LOGGER.setLevel('ERROR')

    def get_best_detection(self, image):
        """
        Get the highest confidence detection from an image that meets area threshold.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            dict: Detection information or None if no valid detection
        """
        # Convert BGR to RGB for YOLO
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Run YOLO detection
        results = self.model(img_rgb, device=self.device)
        
        # Find highest confidence detection that meets area threshold
        highest_conf = 0
        best_detection = None
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                conf = float(box.conf)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                
                # Only consider detection if it meets area threshold
                if area >= self.area_threshold and conf > highest_conf:
                    highest_conf = conf
                    best_detection = {
                        'class_name': self.model.names[int(box.cls)],
                        'confidence': conf,
                        'box': (x1, y1, x2, y2),
                        'area': area
                    }
        
        # Only return detection if confidence exceeds threshold
        if best_detection and best_detection['confidence'] >= self.conf_threshold:
            return best_detection
        return None

def main():
    # Initialize detector
    detector = DetectionInterface(
        model_path="runs/detect/train2/weights/best.pt",
        conf_threshold=0.95,
        area=1500
    )
    
    # Initialize robot and camera (assuming PiBot setup remains the same)
    script_path = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.abspath(os.path.join(script_path, "../PenguinPi-robot/software/python/client/")))
    from pibot_client import PiBot
    bot = PiBot(ip='192.168.1.179')
    
    try:
        while True:
            # Get and preprocess image
            img = bot.getImage()
            img_cropped = img[120:, :]
            img_resized = cv2.resize(img_cropped, (120, 120), interpolation=cv2.INTER_AREA)
            
            # Get best detection
            detection = detector.get_best_detection(img_resized)
            
            if detection:
                print("\nDetection found:")
                print(f"Object: {detection['class_name']}")
                print(f"Confidence: {detection['confidence']:.3f}")
                print(f"Area: {detection['area']} pixels")
                input("Press Enter to continue...")
                
            #     # Draw detection on image
            #     x1, y1, x2, y2 = detection['box']
            #     cv2.rectangle(img_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
            #     # Add label
            #     label = f"{detection['class_name']} {detection['confidence']:.2f}"
            #     (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            #     cv2.rectangle(img_resized, (x1, y1-20), (x1+w, y1), (0, 255, 0), -1)
            #     cv2.putText(img_resized, label, (x1, y1-5), 
            #                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # # Display image
            # cv2.imshow('Detections', img_resized)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
                
    finally:
        cv2.destroyAllWindows()
        bot.setVelocity(0, 0)

if __name__ == "__main__":
    main()
