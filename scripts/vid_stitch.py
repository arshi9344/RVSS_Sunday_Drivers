import os
from PIL import Image
import cv2
import numpy as np
from datetime import datetime

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

def create_video_from_images(input_folder, output_file, fps=60, max_duration=5):
    # Get all jpg files
    image_files = [f for f in os.listdir(input_folder) if f.endswith('.jpg')]
    if not image_files:
        print(f"No images found in {input_folder}")
        return
        
    # Sort by frame number
    image_files.sort(key=get_frame_number)
    
    # Limit frames based on max duration
    max_frames = fps * max_duration
    if len(image_files) > max_frames:
        image_files = image_files[:max_frames]
        print(f"Limiting video to {max_duration} seconds ({max_frames} frames)")
    
    # Create video as before
    first_image = cv2.imread(os.path.join(input_folder, image_files[0]))
    height, width = first_image.shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
    # Process each image
    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        frame = cv2.imread(image_path)
        if frame is not None:
            out.write(frame)
    
    out.release()
    print(f"Video created successfully: {output_file}")

# Example usage
input_folder = "./data/train/20250206_082018/"
output_file = "output_video.mp4"
create_video_from_images(input_folder, output_file)