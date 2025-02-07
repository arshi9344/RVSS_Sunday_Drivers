#!/usr/bin/env python3
import time
import os
import cv2
import numpy as np
import argparse
import matplotlib
matplotlib.use("tkagg")  # or another backend if desired
from machinevisiontoolbox import Image
from robot_comms import get_bot, RobotController

# Define the script path so images can be saved appropriately
script_path = os.path.dirname(os.path.realpath(__file__))

def detect_stop_sign_and_draw(frame_bgr, area_min=200, area_max=500):
    """
    Given a BGR frame from the PiBot's camera, do:
     - Crop
     - Convert to HSV
     - Threshold red
     - Morphological open
     - Blob detection (MVT)
     - Draw bounding boxes on a copy of the mask for visualization
    Returns:
      - detected: bool, True if a blob's area is between area_min and area_max
      - display_mask: an 8-bit image with bounding boxes drawn (for display)
    """

    # 1. Crop to bottom half
    h, w = frame_bgr.shape[:2]
    frame_cropped = frame_bgr[h//2:, :]

    # 2. Convert to HSV
    hsv = cv2.cvtColor(frame_cropped, cv2.COLOR_BGR2HSV)

    # 3. Threshold for red (tweak as needed)
    lower_red = np.array([165, 50, 50])
    upper_red = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # 4. Light morphological opening
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 5. Wrap for MVT blob detection
    mask_bool = mask_clean.astype(bool)
    mvt_im = Image(mask_bool)

    detected = False

    # Prepare a 3-channel image to draw colored boxes (yellow rectangles, etc.)
    display_mask = cv2.cvtColor(mask_clean, cv2.COLOR_GRAY2BGR)

    try:
        blobs = mvt_im.blobs()
        for b in blobs:
            if area_min <= b.area <= area_max:
                detected = True
                print(f"STOP SIGN DETECTED OF AREA {b.area:.1f} (within [{area_min}, {area_max}])")
            # Draw bounding box around every blob for visualization
            rmin, cmin, rmax, cmax = b.bbox
            color = (0, 255, 255)  # yellow for blobs outside range
            if area_min <= b.area <= area_max:
                color = (0, 255, 0)  # green for blobs within range
            cv2.rectangle(display_mask, (cmin, rmin), (cmax, rmax), color, 2)
    except ValueError:
        # no blobs found
        pass

    return detected, display_mask

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="localhost", help="PiBot IP address")
    parser.add_argument("--test", action="store_true", help="Test mode (no real robot)")
    parser.add_argument("--debug", action="store_true", help="Debug prints")
    args = parser.parse_args()

    # Connect to robot
    bot = get_bot(test_mode=args.test, ip=args.ip, debug=args.debug)
    robot = RobotController(bot, debug=args.debug)
    robot.start()

    # Optionally stop the robot initially
    robot.queue_command(0, 0)

    print("Press Ctrl+C to quit.")

    try:
        # Add counter for saved images
        save_counter = 0

        while True:
            # Grab the latest image from the PiBot queue
            frame_bgr = robot.image_queue.get()
            if frame_bgr is None:
                continue

            # Show the raw camera feed in one window
            cv2.imshow("Camera Feed", frame_bgr)

            # Run the blob detection
            found_stop, mask_vis = detect_stop_sign_and_draw(frame_bgr, area_min=200, area_max=500)

            # Show the thresholded mask + bounding boxes
            cv2.imshow("Blob Detection", mask_vis)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC to quit
                break
            elif key == ord('s'):  # 'S' key to save images
                # Create 'saved_images' directory if it doesn't exist
                save_dir = os.path.join(script_path, "saved_images")
                os.makedirs(save_dir, exist_ok=True)
                
                # Save both original and mask images
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                orig_filename = os.path.join(save_dir, f"image_{timestamp}_{save_counter}.jpg")
                mask_filename = os.path.join(save_dir, f"mask_{timestamp}_{save_counter}.jpg")
                
                cv2.imwrite(orig_filename, frame_bgr)
                cv2.imwrite(mask_filename, mask_vis)
                
                print(f"Saved images to {orig_filename} and {mask_filename}")
                save_counter += 1

            # Optional: if found_stop, you could do something
            # e.g. robot.queue_command(0,0) to stop
    except KeyboardInterrupt:
        pass
    finally:
        # cleanup
        robot.queue_command(0, 0)
        robot.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
