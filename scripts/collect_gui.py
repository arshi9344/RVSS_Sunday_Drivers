#!/usr/bin/env python
import time
import sys
import os
import cv2
import numpy as np
import argparse
import signal
import pygame


from pynput import keyboard  # if needed for extra global key control


# Add client path so we can import PiBot
script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_path, "../PenguinPi-robot/software/python/client/")))
from pibot_client import PiBot


parser = argparse.ArgumentParser(description='GUI PiBot controller with live camera view')
parser.add_argument('--ip', type=str, default='localhost', help='IP address of PiBot')
parser.add_argument('--folder', type=str, default='train', help='Folder to save captured images')
parser.add_argument('--im_num', type=int, default=0, help='Initial image number')
parser.add_argument('--display', action='store_true', help="Display pygame window")
args = parser.parse_args()


save_folder = os.path.join(script_path, "../data/", args.folder)
if not os.path.exists(save_folder):
    print(f'Folder "{args.folder}" does not exist in path {save_folder}. Please create it.')
    sys.exit(1)


bot = PiBot(ip=args.ip)
bot.setVelocity(0, 0)


# Conditionally initialize pygame if display is enabled
if args.display:
    pygame.init()
    window_size = (640, 480)
    screen = pygame.display.set_mode(window_size)
    pygame.display.set_caption("RVSS GUI")
    clock = pygame.time.Clock()
else:
    screen = None  # No display; video feed will not be shown


# Global variables for control
angle = 0.0
speed = 0.1
continue_running = True
im_number = args.im_num


def signal_handler(sig, frame):
    global continue_running
    print("\nCtrl+C detected. Stopping robot and exiting gracefully...")
    bot.setVelocity(0, 0)
    continue_running = False
    pygame.quit()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


# Optional: Additional keyboard handling with pynput if needed
def on_press(key):
    global angle, continue_running
    try:
        if key == keyboard.Key.up:
            angle = 0
            print("straight")
        elif key == keyboard.Key.down:
            angle = 0
        elif key == keyboard.Key.right:
            print("right")
            angle += 0.1
        elif key == keyboard.Key.left:
            print("left")
            angle -= 0.1
        elif key == keyboard.Key.space:
            print("stop")
            bot.setVelocity(0, 0)
            # continue_running = False
    except Exception as e:
        print(f"Error: {e}")
        bot.setVelocity(0, 0)


listener = keyboard.Listener(on_press=on_press)
listener.start()


robot_stopped = False  # Flag to pause robot movement
direction = 1          # 1 for forward; -1 for reverse


def adjust_speed(Kd_base, angle):
    # Computes a reduction factor: when angle is 0, factor is 1.
    # Maximum reduction with an angle of 0.5.
    factor = 1 - (abs(angle) * 0.8)
    return int(Kd_base * factor)


# Main loop
try:
    while continue_running:
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                continue_running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    print("stop")
                    bot.setVelocity(0, 0)
                    robot_stopped = True
                else:


                    if robot_stopped:
                        robot_stopped = False
                    if event.key == pygame.K_UP:
                        direction = 1
                        angle = 0
                        print("forward, straight")
                    elif event.key == pygame.K_DOWN:
                        direction = -1
                        angle = 0
                        print("reverse, straight")
                    elif event.key == pygame.K_RIGHT:
                        print("right")
                        angle += 0.1
                    elif event.key == pygame.K_LEFT:
                        print("left")
                        angle -= 0.1


        # Get image from robot
        img = bot.getImage()
        if img is None:
            continue


        # Apply movement logic only if robot is not stopped
        if not robot_stopped:
            angle = np.clip(angle, -0.5, 0.5)
            Kd_base = 30  # Base wheel speed
            Ka = 30  # Turn speed
            adjusted_Kd = adjust_speed(Kd_base, angle)
            left  = int(direction * (adjusted_Kd + Ka * angle))
            right = int(direction * (adjusted_Kd - Ka * angle))
            bot.setVelocity(left, right)


        # Update display if enabled
        if screen is not None:
            frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.resize(frame_rgb, window_size)
            surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
            screen.blit(surface, (0, 0))
            pygame.display.flip()


            # Process pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            clock.tick(30)


    # Clean up on exit
    bot.setVelocity(0, 0)
    listener.stop()
    pygame.quit()
    print("Script ended")


except KeyboardInterrupt:
    bot.setVelocity(0, 0)
    listener.stop()
    pygame.quit()
    print("\nKeyboardInterrupt caught. Exiting gracefully.")
