#!/usr/bin/env python3
import time
import cv2
import os
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
parser.add_argument('--debug_images', action='store_true', help='Show debug image windows')
args = parser.parse_args()

def detect_stop_sign(image, min_area=200, max_area=500):
    """
    Detect a stop sign in the given image based on red color blob detection.
    Returns True if a blob's area is between min_area and max_area.
    """
    print("[DEBUG] Original image shape:", image.shape)
    # Add horizontal cropping while keeping vertical crop
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
            if b.area > sign_area_min and b.area < sign_area_max:
                print("[DEBUG] Blob area exceeds threshold, stop sign detected.")
                return True
        print("[DEBUG] No blob exceeded the sign_area_min threshold.")
        return False
    except ValueError:
        print("[DEBUG] Blob detection raised a ValueError.")
        return False
    
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
model.load_state_dict(torch.load('best_model.pth', map_location=torch.device('cpu'),weights_only=True))
model.eval()

# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

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
        
        if detect_stop_sign(im):
            print("STOP SIGN DETECTED!")
            # Immediately stop the robot if stop sign detected
            robot.queue_command(0, 0)
            time.sleep(2)
        else:
            if args.debug:
                print("[DEBUG] No stop sign detected.")
        
        # Apply CLAHE
        im_lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
        im_lab[:, :, 0] = clahe.apply(im_lab[:, :, 0])
        im = cv2.cvtColor(im_lab, cv2.COLOR_LAB2RGB)
        if args.debug:
            print(f"[DEBUG] applied image transform:")

        # Convert to tensor and normalize
        im_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((120, 120)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])(im).unsqueeze(0)

        im_tensor = im_tensor.to(device)

        if args.debug:
            print(f"[DEBUG] Running model with: {im_tensor.shape}")

        # Model inference
        with torch.no_grad():
            output_tensor = model(im_tensor)
            if args.debug:
                print(f"[DEBUG] Raw model output: {output_tensor}")
            # Transfer to CPU before converting to NumPy
            speeds = output_tensor[0].cpu().numpy() * 60.0  # Denormalize speeds
            speeds = np.clip(speeds, -60, 60)
            speeds = speeds.astype(int)

        if args.debug:
            print(f"[DEBUG] Calculated Speeds: {speeds}")
        # Queue commands at fixed rate
        if current_time - last_command_time >= command_interval:
            robot.queue_command(*speeds)
            #if last_speeds is None or not np.array_equal(speeds, last_speeds):
                
                #last_speeds = speeds
            last_command_time = current_time

except KeyboardInterrupt:
    robot.queue_command(0, 0)
    robot.stop()
finally:
    robot.queue_command(0, 0)
    robot.stop()
    if args.debug:
        print("[DEBUG] Shutting down")
