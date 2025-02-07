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
import torch.func
from model import Net

script_path = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser(description='PiBot client')
parser.add_argument('--ip', type=str, default='localhost', help='IP address of PiBot')
parser.add_argument('--test', action='store_true', default=False, help='Run in test mode')
parser.add_argument('--debug', action='store_true', help='Enable debug mode')
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
            if b.area > min_area and b.area < max_area:
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

# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# After model definition, load multiple models into a list
models = []
models_dir = os.path.join(os.path.dirname(__file__), "ADAM_Models")
model_files = sorted([f for f in os.listdir(models_dir) if f.endswith('.pth')])

if not model_files:
    raise FileNotFoundError(f"No .pth model files found in {models_dir}")

model_files = ['best_model_10.pth', 'best_model_20.pth', 'best_model_30.pth']

for model_file in model_files:
    model = Net()
    model.load_state_dict(torch.load(model_file, map_location=device, weights_only=True))
    model.eval()
    model.to(device)
    models.append(model)

# Create a batched model function using vmap
def model_forward(model_params, input_data):
    return torch.func.functional_call(model, model_params, (input_data,))

# Extract model parameters
model_params = [dict(model.named_parameters()) for model in models]

# Create vectorized version that runs all models in parallel
batched_forward = torch.func.vmap(model_forward, in_dims=(0, None))

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
        
        # if detect_stop_sign(im):
        #     print("STOP SIGN DETECTED!")
        #     # Immediately stop the robot if stop sign detected
        #     robot.queue_command(0, 0)
        #     time.sleep(2)
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

        # Vectorized model ensemble inference
        with torch.no_grad():
            # Run all models in parallel using vmap
            outputs = batched_forward(model_params, im_tensor)
            
            # Average predictions
            ensemble_output = torch.mean(outputs, dim=0)
            speeds = ensemble_output[0].cpu().numpy() * 60.0
            speeds = np.clip(speeds, 0, 60)
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
