#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from machinevisiontoolbox import *
from matplotlib.ticker import ScalarFormatter
from matplotlib import cm
import cv2

def main():
    # RPI Camera v2.1 specs
    SENSOR_WIDTH = 3.68  # mm
    SENSOR_HEIGHT = 2.76 # mm
    FOCAL_LENGTH = 3.04  # mm

    # Load and crop image
    im = cv2.imread('input.jpg')
    h, w = im.shape[:2]
    im = im[120:h, :]
    print(im)

    # Source points
    p1 =   np.array([
    [44.1364,  377.0654], 
    [94.0065,  152.7850],
    [537.8506,  163.4019],
    [611.8247,  366.4486]
    ])

    mn = p1.min(axis=1)
    mx = p1.max(axis=1)
    
    # Destination points
    p2 = np.array([
    [mn[0], mn[0], mx[0], mx[0]],
    [mx[1], mn[1], mn[1], mx[1]]
    ])

    # Create camera model with RPI v2.1 parameters
    cam = CentralCamera(
        imagesize=[w, h],
        f=FOCAL_LENGTH/1000,  # Convert mm to meters
        sensorsize=[SENSOR_WIDTH/1000, SENSOR_HEIGHT/1000]  # Convert mm to meters
    )

    H, _ = CentralCamera.points2H(p1, p2, method='leastsquares')

    warped = im.warp_perspective(H)
    warped.disp(grid=True)

if __name__ == "__main__":
    main()