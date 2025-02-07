#!/usr/bin/env python
import sys
import os
import numpy as np
import pygame
import cv2
from PIL import Image
from ultralytics import YOLO
import torch

script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_path, "../PenguinPi-robot/software/python/client/")))
from pibot_client import PiBot

# Initialize robot
bot = PiBot(ip='192.168.1.179')
bot.setVelocity(0, 0)

# Initialize YOLO model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
model = YOLO("runs/detect/train10/weights/best.pt")
model.to(device)

# Set confidence threshold
CONF_THRESHOLD = 0.5

# Suppress YOLO printing
from ultralytics.utils import LOGGER
LOGGER.setLevel('ERROR')  # Only show error messages

# Initialize Pygame for keyboard input
pygame.init()
pygame.display.set_mode((100, 100))

running = True
paused = False
while running:
    if not paused:
        # Get image from bot
        img = bot.getImage()
        
        # Crop 120 pixels from top and bottom
        img_cropped = img[120:, :]
        
        # Resize to 120x120
        img_resized = cv2.resize(img_cropped, (120, 120), interpolation=cv2.INTER_AREA)
        
        # Convert BGR to RGB for YOLO
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # Run YOLO detection with confidence threshold
        results = model(img_rgb, device=device, conf=CONF_THRESHOLD)
        
        # Find highest confidence detection
        detected = False
        highest_conf = 0
        best_box = None
        best_cls_name = None
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                conf = float(box.conf)
                if conf > highest_conf:
                    highest_conf = conf
                    best_box = box.xyxy[0]
                    best_cls_name = model.names[int(box.cls)]
                    detected = True
        
        # Draw only the highest confidence box
        if detected and highest_conf > 0.9:
            x1, y1, x2, y2 = map(int, best_box)
            
            # Calculate box area
            box_area = (x2 - x1) * (y2 - y1)
            
            # Draw rectangle and label
            cv2.rectangle(img_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            label = f'{best_cls_name} {highest_conf:.2f}'
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img_resized, (x1, y1-20), (x1+w, y1), (0, 255, 0), -1)
            cv2.putText(img_resized, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Print detection information
            print(f"Detected {best_cls_name}")
            print(f"Confidence: {highest_conf:.3f}")
            print(f"Bounding box area: {box_area} pixels")
            
            paused = True
            print("Press SPACE to continue...")
    
    # Display the processed image with detections
    cv2.imshow('Robot Camera with YOLO Detection', img_resized)
    key = cv2.waitKey(1)
    
    # Handle Pygame events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                running = False
            elif event.key == pygame.K_SPACE and paused:
                paused = False
                print("Continuing...")

# Clean up
cv2.destroyAllWindows()
pygame.quit()
bot.setVelocity(0, 0)
