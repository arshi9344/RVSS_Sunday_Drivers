import os
import sys
import cv2

script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_path, "../PenguinPi-robot/software/python/client/")))

from pibot_client import PiBot

# Create a connection to the PiBot
bot = PiBot("192.168.1.179")

# Request an image from the bot
image = bot.getImage()

# Define the subfolder name for saving images
subfolder = "images"

# Get the absolute path to the subfolder inside the current directory
folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), subfolder)

# Create the subfolder if it does not exist
if not os.path.exists(folder):
    os.makedirs(folder)

# List files in the subfolder matching the naming pattern "image_*.jpg" (case insensitive)
existing_files = [f for f in os.listdir(folder)
                  if f.lower().startswith("image_") and f.lower().endswith(".jpg")]

# Determine the highest existing index from the filenames
max_index = 0
for f in existing_files:
    try:
        index = int(f.split('_')[1].split('.')[0])
        if index > max_index:
            max_index = index
    except (IndexError, ValueError):
        continue

# Set the next index and build the new filename
new_index = max_index + 1
new_filename = f"image_{new_index}.jpg"
filepath = os.path.join(folder, new_filename)

# Save the image
cv2.imwrite(filepath, image)
print(f"Saved new image: {filepath}")

# Close the connection
bot.stop()