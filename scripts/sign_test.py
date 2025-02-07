#!/usr/bin/env python3
import cv2
import numpy as np
import argparse
import time
from machinevisiontoolbox import Image

import matplotlib
matplotlib.use("tkagg")  # or whichever backend works for your environment

from robot_comms import get_bot, RobotController

def detect_stop_sign_with_crop(frame_bgr, sign_area_min=2000):
    """
    Follows the logic from 'stop_blob.py':
      1. Crop bottom half + horizontal slice
      2. Convert to HSV
      3. Threshold for red [165..180, 50..220, 50..220]
      4. Morphological open
      5. Blob detection with MVT
    Returns:
      - detected: bool (True if any blob area > sign_area_min)
      - overlay:  BGR image for visualization of the mask and bounding boxes
    """

    # -------------------------------------------------
    # STEP 1: Crop (bottom half + horizontal slice)
    # -------------------------------------------------
    h, w = frame_bgr.shape[:2]
    crop_vertical = h // 2         # bottom half
    crop_horizontal = w // 6       # remove ~1/6 from each side
    # Crop vertically, then horizontally
    frame_cropped = frame_bgr[crop_vertical:, crop_horizontal : w - crop_horizontal]
    ch, cw = frame_cropped.shape[:2]  # for reference

    # -------------------------------------------------
    # STEP 2: Convert to HSV + Threshold
    # -------------------------------------------------
    hsv = cv2.cvtColor(frame_cropped, cv2.COLOR_BGR2HSV)
    
    # Red range from stop_blob.py
    lower_red = np.array([165,  50,  50])
    upper_red = np.array([180, 220, 220])
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # -------------------------------------------------
    # STEP 3: Morphological open (to keep holes for letters)
    # -------------------------------------------------
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask_clean_cv = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # -------------------------------------------------
    # STEP 4: Blob detection
    # -------------------------------------------------
    mask_bool = mask_clean_cv.astype(bool)
    mvt_image = Image(mask_bool)
    
    detected = False
    # For drawing bounding boxes on the mask
    overlay = cv2.cvtColor(mask_clean_cv, cv2.COLOR_GRAY2BGR)

    try:
        blobs = mvt_image.blobs()
        for b in blobs:
            if b.area > sign_area_min:
                detected = True
            
            # Draw bounding box on overlay
            (rmin, cmin, rmax, cmax) = b.bbox
            color = (0, 255, 255)  # yellow
            if b.area > sign_area_min:
                color = (0, 255, 0)  # green for "big" blob
            cv2.rectangle(overlay, (cmin, rmin), (cmax, rmax), color, 2)
    except ValueError:
        # no blobs found
        pass

    return detected, overlay

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="localhost", help="IP of PiBot")
    parser.add_argument("--test", action="store_true", help="Test mode (no real robot)")
    parser.add_argument("--debug", action="store_true", help="Debug prints")
    args = parser.parse_args()

    # Setup PiBot connection
    bot = get_bot(test_mode=args.test, ip=args.ip, debug=args.debug)
    robot = RobotController(bot, debug=args.debug)
    robot.start()
    robot.queue_command(0,0)  # ensure robot is stopped initially

    print("Press Ctrl+C to exit.")

    try:
        while True:
            frame_bgr = robot.image_queue.get()  # get the next camera frame
            if frame_bgr is None:
                continue  # no frame yet

            # Show the raw camera feed
            cv2.imshow("Camera Feed", frame_bgr)

            # Perform the custom stop-sign detection logic
            found_sign, mask_overlay = detect_stop_sign_with_crop(frame_bgr, sign_area_min=2000)

            if found_sign:
                print("STOP SIGN DETECTED!")
                # You could do something here, like stop the robot
                # robot.queue_command(0, 0)

            # Show the thresholded mask overlay
            cv2.imshow("Blob Mask (Cropped + Red Threshold)", mask_overlay)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC to quit
                break

    except KeyboardInterrupt:
        pass
    finally:
        robot.queue_command(0,0)
        robot.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
