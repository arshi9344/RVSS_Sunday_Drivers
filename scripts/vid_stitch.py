import os
from PIL import Image
import cv2
import numpy as np
from datetime import datetime
import tkinter as tk
from tkinter import filedialog

def get_timestamp_from_filename(filename):
    # Assuming filename format contains timestamp (modify pattern as needed)
    try:
        # Remove extension and try to parse date
        name = os.path.splitext(filename)[0]
        return datetime.strptime(name, "%Y%m%d_%H%M%S")
    except:
        return None

def get_frame_number(filename):
    """Extract frame number from filename format: 000001_angle_left_right.jpg"""
    return int(filename.split('_')[0])

def get_speeds_from_filename(filename):
    """Extract left/right speeds from filename: 000001_angle_left_right.jpg"""
    parts = filename.split('_')
    left = float(parts[-2])
    right = float(parts[-1].split('.jpg')[0])
    return left, right

def create_video_from_images(input_folder, output_file, fps=30, max_duration=None):
    # Get all jpg files
    image_files = [f for f in os.listdir(input_folder) if f.endswith('.jpg')]
    if not image_files:
        print(f"No images found in {input_folder}")
        return
        
    # Sort by frame number
    image_files.sort(key=get_frame_number)
    
    # Handle max_duration
    if max_duration is not None:
        max_frames = fps * max_duration
        image_files = image_files[:max_frames]
        print(f"Limiting video to {max_duration} seconds ({max_frames} frames)")
    
    # Create video
    first_image = cv2.imread(os.path.join(input_folder, image_files[0]))
    height, width = first_image.shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
    # Process each image
    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        frame = cv2.imread(image_path)
        
        if frame is not None:
            # Get speeds from filename
            left_speed, right_speed = get_speeds_from_filename(image_file)
            
            # Add text overlay
            text = f"L: {left_speed:.1f} R: {right_speed:.1f}"
            cv2.putText(frame, text, 
                       (10, height-20), # Position at bottom left
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, # Font scale
                       (255,255,255), # White color
                       2) # Thickness
            
            out.write(frame)
    
    out.release()
    print(f"Video created successfully: {output_file}")

def select_folder():
    root = tk.Tk()
    root.withdraw() # Hide main window
    
    # Start in the ./data/train directory
    initial_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data", "train")
    
    folder = filedialog.askdirectory(
        initialdir=initial_dir,
        title="Select folder containing images"
    )
    print(f"Selected folder: {folder}")
    if folder:
        # Get folder name for output file
        folder_name = os.path.basename(folder)
        output_file = f"{folder_name}.mp4"
        
        # Create video
        create_video_from_images(folder, output_file)
    else:
        print("No folder selected")

if __name__ == "__main__":
    select_folder()