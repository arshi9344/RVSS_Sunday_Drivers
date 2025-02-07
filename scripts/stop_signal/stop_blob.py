import cv2
import numpy as np
import matplotlib.pyplot as plt
from machinevisiontoolbox import Image 

# -------------------------------------------------
# STEP 1: Load & Crop
# -------------------------------------------------
cv_frame = cv2.imread('./scripts/image_1.jpg')
if cv_frame is None:
    raise FileNotFoundError("Could not load './scripts/image_1.jpg'!")
# Make a copy of the original to overlay blobs later
overlay_img = cv_frame.copy()

# For processing, crop to bottom half
h, w = cv_frame.shape[:2]
cv_frame_cropped = cv_frame[h//2:, :]

# Convert cropped image from BGR -> RGB for display + MVT
cv_frame_rgb = cv2.cvtColor(cv_frame_cropped, cv2.COLOR_BGR2RGB)

# -------------------------------------------------
# STEP 2: Threshold for Red, Aiming to Keep Letters as Holes
# -------------------------------------------------
hsv = cv2.cvtColor(cv_frame_cropped, cv2.COLOR_BGR2HSV)
lower_red = np.array([165,  50,  50])
upper_red = np.array([180, 255, 255])
mask = cv2.inRange(hsv, lower_red, upper_red)

# -------------------------------------------------
# STEP 3: LIGHT Morphological Operation
# -------------------------------------------------
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
mask_clean_cv = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask_mvt = Image(mask_clean_cv.astype(bool))

# -------------------------------------------------
# STEP 4: Blob Detection with MVT and Overlay on Original Image
# -------------------------------------------------
sign_area_min = 300
sign_area_max = 500  # Optional maximum threshold if needed

try:
    blobs = mask_mvt.blobs()
    print(f"Number of blobs found: {len(blobs)}")

    # Overlay bounding boxes and centroids on the original image.
    # Note: Blob coordinates are relative to the cropped image.
    # Therefore, add an offset of h//2 to the row components.
    for i, b in enumerate(blobs):
        print(f"Blob {i}: area={b.area}, bbox={b.bbox}, centroid={b.centroid}")
        if b.area > sign_area_min and b.area < sign_area_max:
            # Adjust bounding box from cropped to original image coordinates
            min_row, min_col, max_row, max_col = b.bbox
            min_row += h // 2
            max_row += h // 2
            cv2.rectangle(overlay_img, (min_col, min_row), (max_col, max_row), (0, 255, 0), 2)
            # Adjust centroid and draw it
            c_row, c_col = b.centroid
            c_row = int(c_row + h // 2)
            c_col = int(c_col)
            cv2.circle(overlay_img, (c_col, c_row), 3, (0, 0, 255), -1)
            print("STOPPING !")
    
    # Display the overlay image with blob detections in a new figure using matplotlib
    plt.figure(figsize=(10, 8))
    overlay_rgb = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)
    plt.imshow(overlay_rgb)
    plt.title("Stop Sign Blob Overlays on Original Image")
    plt.axis('off')
    plt.show()

except ValueError:
    print("No blobs found—check your thresholds or morphological operations.")