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
from torch.func import stack_module_state, functional_call
import copy
import torch.func
from model import Net
# Add import for detection
from detection import DetectionInterface

script_path = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser(description='PiBot client')
parser.add_argument('--ip', type=str, default='localhost', help='IP address of PiBot')
parser.add_argument('--test', action='store_true', default=False, help='Run in test mode')
parser.add_argument('--debug', action='store_true', help='Enable debug mode')
args = parser.parse_args()

# Add at start of main script, after argument parsing:
last_stop_time = 0
STOP_COOLDOWN = 3  # seconds

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

# Load models
models = []
models_dir = ("ADAM_Models")
model_files = sorted([f for f in os.listdir(models_dir) if f.endswith('.pth')])

if args.debug:
    print(f"[DEBUG] Found model files: {model_files}")

for model_file in model_files:
    if args.debug:
        print(f"[DEBUG] Loading model: {model_file}")
    model = Net()
    model_path = os.path.join(models_dir, model_file)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    models.append(model)

if args.debug:
    print(f"[DEBUG] Loaded {len(models)} models")

# Create stateless base model and stack states
base_model = copy.deepcopy(models[0])
base_model = base_model.to('meta')
params, buffers = stack_module_state(models)

if args.debug:
    print("[DEBUG] Stacked parameter shapes:")
    for k, v in params.items():
        print(f"  {k}: {v.shape}")

# Define forward function for vmap
def fmodel(params, buffers, x):
    return functional_call(base_model, (params, buffers), (x,))

# Create vectorized version
batched_forward = torch.vmap(fmodel, in_dims=(0, 0, None))

# Initialize CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

# After device initialization, add detector initialization
detector = DetectionInterface(
    model_path=os.path.join(os.path.dirname(__file__), "best.pt"),
    conf_threshold=0.95,
    area=1500
)

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
        
        # Check for stop sign
        detection = detector.get_best_detection(im)
        current_time = time.time()
        if detection and detection['class_name'] == 'sign':
            # Get bounding box coordinates
            x1, y1, x2, y2 = detection['box']
            
            # Check if box is in bottom third and cooldown elapsed
            image_height = im.shape[0]
            min_y = 2 * image_height // 3
            
            if y2 > min_y and (current_time - last_stop_time) > STOP_COOLDOWN:
                print("STOP SIGN DETECTED!")
                robot.queue_command(0, 0)
                time.sleep(2)
                last_stop_time = current_time
                if args.debug:
                    print(f"[DEBUG] Stop sign detected with confidence: {detection['confidence']:.3f}")
                    print(f"[DEBUG] Box position: y2={y2}, threshold={min_y}")
            elif args.debug:
                if (current_time - last_stop_time) <= STOP_COOLDOWN:
                    print("[DEBUG] Stop sign ignored - cooldown active")
                else:
                    print("[DEBUG] Sign detected but not in bottom third of image")
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

        # In the main loop, replace inference section with:
        with torch.no_grad():
            if args.debug:
                print("[DEBUG] Running inference with:")
                print(f"  Input shape: {im_tensor.shape}")
            
            # Run all models in parallel
            outputs = batched_forward(params, buffers, im_tensor)
            
            if args.debug:
                print("[DEBUG] Raw outputs shape:", outputs.shape)
                print("[DEBUG] Individual model outputs:", outputs.cpu().numpy())
            
            # Average predictions
            ensemble_output = torch.mean(outputs, dim=0)
            if args.debug:
                print("[DEBUG] Averaged output:", ensemble_output.cpu().numpy())
            speeds = ensemble_output[0].cpu().numpy() * 60
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
