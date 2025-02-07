import cv2
import numpy as np
import matplotlib.pyplot as plt
from machinevisiontoolbox import Image 

# -------------------------------------------------
# STEP 1: Load & Crop
# -------------------------------------------------
image_path='./images/image_4.jpg'
cv_frame = cv2.imread(image_path)
if cv_frame is None:
    raise FileNotFoundError(f"Could not load {image_path}!")

# Crop to bottom half and middle portion
h, w = cv_frame.shape[:2]
crop_vertical = h//2  # Keep bottom half
crop_horizontal = w//6  # Cut 25% from each side

# Crop vertically first, then horizontally
cv_frame_cropped = cv_frame[crop_vertical:, crop_horizontal:w-crop_horizontal]

# Convert cropped image from BGR -> RGB for display + MVT
cv_frame_rgb = cv2.cvtColor(cv_frame_cropped, cv2.COLOR_BGR2RGB)

# Display the cropped RGB image
plt.figure(figsize=(10, 6))
plt.imshow(cv_frame_rgb)
plt.title('Cropped RGB Image (Bottom Half + Middle Portion)')
plt.axis('on')  # Keep axis to see dimensions
plt.show()

# -------------------------------------------------
# STEP 2: Threshold for Red, Aiming to Keep Letters as Holes
# -------------------------------------------------
# Convert cropped BGR to HSV for color thresholding
hsv = cv2.cvtColor(cv_frame_cropped, cv2.COLOR_BGR2HSV)

# Example red range (tweak as needed). 
# The idea is that the STOP sign (red) becomes white (255) in the mask.
lower_red = np.array([165,  50,  50])
upper_red = np.array([180, 220, 220])
mask = cv2.inRange(hsv, lower_red, upper_red)

# -------------------------------------------------
# STEP 3: LIGHT Morphological Operation
# -------------------------------------------------
# We do a small "open" to remove salt noise while
# still allowing the letters to remain black holes.
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
mask_clean_cv = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# Convert from [0,255] to boolean for MVT
mask_mvt = Image(mask_clean_cv.astype(bool))
sign_area_min=2000
# -------------------------------------------------
# STEP 4: Blob Detection with MVT
# -------------------------------------------------
try:
    blobs = mask_mvt.blobs()
    print(f"Number of blobs found: {len(blobs)}")

    # -------------------------------------------------
    # (A) Display in MVT coordinate space
    # -------------------------------------------------
    plt.figure(figsize=(8, 6))
    mask_mvt.disp(title="Thresholded Mask in MVT")
    # Plot bounding boxes + centroids for *all* blobs (parent and children)
    blobs.plot_box(color='yellow', linewidth=2)
    blobs.plot_centroid(color='red', marker='x', markersize=7)
    # Force equal aspect ratio so squares don’t look like rectangles
    plt.axis('equal')
    plt.show()
    print("about to loop through blobs")
    # Print out blob info: parent, children, area, etc.
    for i, b in enumerate(blobs):
        print(f"Blob {i}: area={b.area}, bbox={b.bbox}, parent={b.parent}, #children={len(b.children)}")
        if b.area > sign_area_min: 
            print("STOPPING !")
            
    # -------------------------------------------------
except ValueError:
    print("No blobs found—check your thresholds or morphological ops.")