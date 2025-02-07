#!/usr/bin/env python3
import time
import cv2
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import argparse
import torch.nn.functional as F
import torchvision.transforms as transforms
from robot_comms import RobotController, get_bot
from machinevisiontoolbox import Image  # Required for blob detection

script_path = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser(description='PiBot stop sign detection test')
parser.add_argument('--ip', type=str, default='localhost', help='IP address of PiBot')
parser.add_argument('--test', action='store_true', default=False, help='Run in test mode')
parser.add_argument('--debug', action='store_true', help='Enable debug mode')
args = parser.parse_args()

# Initialize robot using robot_comms
bot = get_bot(test_mode=args.test, ip=args.ip, debug=args.debug)
robot = RobotController(bot, debug=args.debug)
robot.start()

# Stop the robot initially
#robot.queue_command(0, 0)

# Wait for the camera thread to produce an image
if args.debug:
    print("[DEBUG] Waiting for first camera image")
while True:
    im = robot.image_queue.get()
    if im is not None:
        break
    time.sleep(0.1)
if args.debug:
    print("[DEBUG] Got first camera image")

def detect_stop_sign(image, sign_area_min=2000):
    """
    Detect a stop sign in the given image based on red color blob detection.
    Returns True if a blob with area greater than sign_area_min is found.
    """
    print("[DEBUG] Original image shape:", image.shape)
    h, w, _ = image.shape
    cropped = image[h//2:, :]
    print("[DEBUG] After cropping bottom half, cropped shape:", cropped.shape)

    # Convert to HSV for thresholding red
    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
    print("[DEBUG] Converted cropped image to HSV.")
    
    lower_red = np.array([165, 50, 50])
    upper_red = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    print("[DEBUG] Created red threshold mask, mask shape:", mask.shape)
    
    # Morphological operation to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    print("[DEBUG] Applied morphological opening on mask.")
    
    # Convert mask to boolean and use MVT for blob detection
    mask_bool = mask_clean.astype(bool)
    print("[DEBUG] Converted cleaned mask to boolean.")
    
    mvt_im = Image(mask_bool)
    try:
        blobs = mvt_im.blobs()
        print("[DEBUG] Blob detection returned", len(blobs), "blob(s).")
        for b in blobs:
            print("[DEBUG] Blob area:", b.area)
            if b.area > sign_area_min:
                print("[DEBUG] Blob area exceeds threshold, stop sign detected.")
                return True
        print("[DEBUG] No blob exceeded the sign_area_min threshold.")
        return False
    except ValueError:
        print("[DEBUG] Blob detection raised a ValueError.")
        return False

try:
    while True:
        # Use robot_comms image acquisition
        if args.debug:
            print("[DEBUG] Requesting image")
        im = robot.image_queue.get()
        cv2.imwrite("Image_stop.jpg", im)
        input()
        
        if im is None:
            if args.debug:
                print("[DEBUG] No image received")
            continue
        
        if args.debug:
            print(f"[DEBUG] Got image shape: {im.shape}")
        
        # Crop image (simulate removal of irrelevant parts, e.g., top half)
        im = im[120:, :, :]
        if args.debug:
            print(f"[DEBUG] Cropped image shape: {im.shape}")
        
        # Run stop sign detection on the cropped image before further filtering
        if detect_stop_sign(im):
            print("STOP SIGN DETECTED!")
            # Immediately stop the robot if stop sign detected
            #robot.queue_command(0, 0)
        else:
            if args.debug:
                print("[DEBUG] No stop sign detected.")
        
        time.sleep(0.1)
except KeyboardInterrupt:
    print("Exiting...")
finally:
    robot.queue_command(0, 0)
    robot.stop()
    if args.debug:
        print("[DEBUG] Shutting down")
