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
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from robot_comms import RobotController, get_bot

script_path = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser(description='PiBot client')
parser.add_argument('--ip', type=str, default='localhost', help='IP address of PiBot')
parser.add_argument('--test', action='store_true', default=False, help='Run in test mode')
parser.add_argument('--debug', action='store_true', help='Enable debug mode')
parser.add_argument('--debug_images', action='store_true', help='Show debug image windows')
args = parser.parse_args()

# Initialize robot using robot_comms
bot = get_bot(test_mode=args.test, ip=args.ip, debug=args.debug)
robot = RobotController(bot, debug=args.debug)
robot.start()

# Stop the robot
robot.queue_command(0, 0)

#INITIALISE NETWORK HERE
IMAGE_SIZE = (320, 240)
BASE_SPEED = 40
TURN_SPEED = 40

#LOAD NETWORK WEIGHTS HERE
model = Net()
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Initialize CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

#countdown before beginning
print("Get ready...")
time.sleep(1)
print("3")
time.sleep(1)
print("2")
time.sleep(1)
print("1")
time.sleep(1)
print("GO!")

try:
    angle = 0
    is_stopped = bool(False)
    speeds = (0, 0)
    command_interval = 1.0 / 20  # 20Hz command rate

    while True:
        current_time = time.time()
        # Get image from queue
        im = robot.image_queue.get()
        if im is None:
            if args.debug:
                print("[DEBUG] No image received")
            continue
            
        if args.debug:
            print(f"[DEBUG] Got image shape: {im.shape}")
            
        # Process image    
        im = im[120:, :, :]
        if args.debug:
            print(f"[DEBUG] Cropped image shape: {im.shape}")
            
        # # Show debug window
        # if args.debug_images:
        #     cv2.imshow('Raw Image', im)
        #     cv2.waitKey(1)

        # Apply CLAHE
        im_lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
        im_lab[:,:,0] = clahe.apply(im_lab[:,:,0])
        im = cv2.cvtColor(im_lab, cv2.COLOR_LAB2RGB)
        
        # Convert to tensor and normalize
        im_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((60, 60)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])(im).unsqueeze(0)

        # Model inference
        with torch.no_grad():
            output_tensor = model(im_tensor)
            speeds = output_tensor[0].numpy() * 60.0  # Denormalize speeds

        #TO DO: check for stop signs?
        
        # Queue commands at fixed rate
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
