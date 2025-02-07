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

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 1) Convolution + Pooling block
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        
        # 2) Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2)

        # 3) Fully connected
        #    Flattened dimension after conv/pool is 1600, so fc1 in_features=1600
        self.fc1 = nn.Linear(10816, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 2)  # final 2 outputs (e.g. left/right speeds)

        # Optional: You can define activation once and reuse, or inline them in forward
        self.relu = nn.ReLU()

    def forward(self, x):
        # input shape: [batch_size, 3, 60, 60]

        # Conv1 -> ReLU -> Pool => [batch_size, 16, 29, 29]
        x = self.pool(self.relu(self.conv1(x)))
        
        # Conv2 -> ReLU -> Pool => [batch_size, 32, 13, 13]
        x = self.pool(self.relu(self.conv2(x)))
        
        # Conv3 -> ReLU -> Pool => [batch_size, 64, 5, 5]
        x = self.pool(self.relu(self.conv3(x)))

        # Flatten => [batch_size, 64*5*5 = 1600]
        x = torch.flatten(x, start_dim=1)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        out = self.fc3(x)  # shape: [batch_size, 2]

        return out
    
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

# BEFORE the loop starts, e.g. right after printing "GO!"
last_command_time = time.time()
last_speeds = None
command_interval = 0.1  # 10 Hz command rate

try:
    while True:
        current_time = time.time()
        # Get image from queue
        if args.debug:
            print(f"[DEBUG] requesting image")
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

        # Apply CLAHE
        im_lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
        im_lab[:, :, 0] = clahe.apply(im_lab[:, :, 0])
        im = cv2.cvtColor(im_lab, cv2.COLOR_LAB2RGB)
        if args.debug:
            print(f"[DEBUG] applied image transform:")

        # Convert to tensor and normalize
        im_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((60, 60)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])(im).unsqueeze(0)

        #im_tensor = im_tensor.to('gpu')

        if args.debug:
            print(f"[DEBUG] Running model with: {im_tensor.shape}")

        # Model inference
        with torch.no_grad():
            output_tensor = model(im_tensor)
            if args.debug:
                print(f"[DEBUG] Raw model output: {output_tensor}")
            speeds = output_tensor[0].numpy() * 60.0  # Denormalize speeds

        if args.debug:
            print(f"[DEBUG] Calculated Speeds: {speeds}")

        # Queue commands at fixed rate
        if current_time - last_command_time >= command_interval:
            if last_speeds is None or not np.array_equal(speeds, last_speeds):
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
