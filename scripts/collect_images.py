#!/usr/bin/env python
import sys
import os
import numpy as np
import pygame
import cv2
from PIL import Image

script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_path, "../PenguinPi-robot/software/python/client/")))
from pibot_client import PiBot

# Create stopsign directory if it doesn't exist
save_dir = os.path.join(script_path, "stopsign2")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Initialize robot
bot = PiBot(ip='192.168.1.179')
bot.setVelocity(0, 0)

# Initialize Pygame for keyboard input
pygame.init()
# Create a small window (can be minimized, but needed for keyboard input)
pygame.display.set_mode((100, 100))

# Counter for saved images
save_counter = 90

running = True
while running:
    # Get image from bot
    img = bot.getImage()
    
    # Crop 120 pixels from top and bottom (60 from each side)
    img_cropped = img[120:, :]
    
    # Resize to 120x120
    img_resized = cv2.resize(img_cropped, (120, 120), interpolation=cv2.INTER_AREA)
    
    # Display the processed image
    cv2.imshow('Robot Camera', img_resized)
    cv2.waitKey(1)  # Required for the window to update
    
    # Handle Pygame events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_y:
                # Find next available index
                while True:
                    save_path = os.path.join(save_dir, f"image_{save_counter}.jpg")
                    if not os.path.exists(save_path):
                        break
                    save_counter += 1
                
                # Convert to RGB for saving with PIL
                img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                Image.fromarray(img_rgb).save(save_path)
                print(f"Saved image as {save_path}")
                save_counter += 1
                
            elif event.key == pygame.K_q:
                running = False

# Clean up
cv2.destroyAllWindows()
pygame.quit()
bot.setVelocity(0, 0)
