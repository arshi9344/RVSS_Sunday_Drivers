#!/usr/bin/env python3
import time
import click
import math
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

parser = argparse.ArgumentParser(description='PiBot client')
parser.add_argument('--ip', type=str, default='localhost', help='IP address of PiBot')
parser.add_argument('--test', action='store_true', default=False, help='Run in test mode')
parser.add_argument('--debug', action='store_true', help='Enable debug mode')
args = parser.parse_args()

# Initialize robot using robot_comms
bot = get_bot(test_mode=args.test, ip=args.ip, debug=args.debug)
robot = RobotController(bot, debug=args.debug)
robot.start()

# Stop the robot initially
robot.queue_command(0, 0)

# INITIALISE NETWORK HERE
IMAGE_SIZE = (320, 240)
BASE_SPEED = 40
TURN_SPEED = 40
STOP_TIME=1
# LOAD NETWORK WEIGHTS HERE
model = torch.load('best_model.pth')
model.eval()

# Countdown before beginning
print("Get ready...")
time.sleep(1)
print("3")
time.sleep(1)
print("2")
time.sleep(1)
print("1")
time.sleep(1)
print("GO!")

def detect_stop_sign(image, sign_area_min=2000):
    """
    Detect a stop sign in the given image based on red color blob detection.
    Returns True if a blob with area greater than sign_area_min is found.
    """
    # Crop to bottom half
    h, w, _ = image.shape
    cropped = image[h//2:, :]
    
    # Convert to HSV for thresholding red
    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
    lower_red = np.array([165, 50, 50])
    upper_red = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    
    # Morphological operation to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Convert mask to boolean and use MVT for blob detection
    mask_bool = mask_clean.astype(bool)
    mvt_im = Image(mask_bool)
    try:
        blobs = mvt_im.blobs()
        for b in blobs:
            if b.area > sign_area_min:
                return True
        return False
    except ValueError:
        return False

angle = 0
is_stopped = False
speeds = (0, 0)
command_interval = 1.0 / 20  # 20Hz command rate

# Variables to manage command rate
last_command_time = time.time()
last_speeds = (0, 0)

try:
    while True:
        current_time = time.time()
        # Get an image from the robot
        im = robot.image_queue.put(angle, *speeds, is_stopped)
        if im is None:
            continue

        # Check for stop sign in the current image
        if detect_stop_sign(im):
            if args.debug:
                print("[DEBUG] STOP SIGN DETECTED!")
            robot.queue_command(0, 0)  # Immediately stop
            time.sleep(STOP_TIME)              # Pause for 1 second
            is_stopped = False         # Allow driving to resume
            continue                 # Skip processing this iteration
        else:
            is_stopped = False
            # Process image for model
            im_tensor = transforms.ToTensor()(im).unsqueeze(0)
            with torch.no_grad():
                output_tensor = model(im_tensor)
                # Assuming model outputs a steering angle for both wheels
                speeds = (output_tensor[0].item(), output_tensor[0].item())

        # Queue commands at a fixed rate
        if current_time - last_command_time >= command_interval:
            if speeds != last_speeds:
                robot.queue_command(*speeds)
                last_speeds = speeds
            last_command_time = current_time

except KeyboardInterrupt:
    robot.queue_command(0, 0)
    robot.stop()
finally:
    robot.queue_command(0, 0)
    robot.stop()
    if args.debug:
        print("[DEBUG] Shutting down")
