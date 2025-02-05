#!/usr/bin/env python
import sys
import os
import cv2
import pygame
import argparse
from datetime import datetime
import threading
from queue import Queue
import numpy as np
import time
import random

os.environ['SDL_AUDIODRIVER'] = 'dummy'
# os.environ['SDL_VIDEODRIVER'] = 'dummy'  # Comment this out

script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_path, "../PenguinPi-robot/software/python/client/")))

# Import PiBot conditionally 
try:
    from pibot_client import PiBot
except ImportError:
    print("Warning: PiBot client not found, dummy bot will be used")
    PiBot = None

# Parse arguments
parser = argparse.ArgumentParser(description='PiBot client')
parser.add_argument('--ip', type=str, default='localhost', help='IP address of PiBot')
parser.add_argument('--im_num', type=int, default=0)
parser.add_argument('--folder', type=str, default='train')
parser.add_argument('--collect', action='store_true', help='Enable data collection')
parser.add_argument('--test', action='store_true', help='Run in test mode without robot')
args = parser.parse_args()

# Constants
MAX_ANGLE = 0.5

BASE_SPEED = 40
TURN_SPEED = 40
FPS = 30

KEY_REPEAT_DELAY = 100  # ms before key repeat
KEY_REPEAT_INTERVAL = 50  # ms between repeats
RETURN_RATE = 0.08  # Rate of return to center
STEERING_RATE = 0.05

MAX_BOOST_SPEED = 80
BOOST_RATE = 0.2  # Speed increase per frame

CONTROL_FPS = 100  # Main and control loop rate
CAPTURE_FPS = 30   # Image capture rate

command_queue = Queue()
image_queue = Queue()

def handle_input(angle, is_stopped, current_speed):
    try:
        events = pygame.event.get()
        keys = pygame.key.get_pressed()
        
        # Process events
        for event in events:
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and 
                event.key == pygame.K_q):
                return True, angle, is_stopped, current_speed
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                is_stopped = not is_stopped
                if is_stopped:
                    # Clear queue and send stop
                    while not command_queue.empty():
                        command_queue.get_nowait()
                    command_queue.put((0, 0))
        
        # Handle boost
        if keys[pygame.K_UP]:
            current_speed = min(current_speed + BOOST_RATE, MAX_BOOST_SPEED)
        else:
            current_speed = max(current_speed - BOOST_RATE, BASE_SPEED)
        
        # Clear queue before new steering command
        if not is_stopped and (keys[pygame.K_LEFT] or keys[pygame.K_RIGHT]):
            while not command_queue.empty():
                command_queue.get_nowait()
        
        # Update steering
        if keys[pygame.K_LEFT]:
            angle = max(angle - STEERING_RATE, -MAX_ANGLE)
        elif keys[pygame.K_RIGHT]:
            angle = min(angle + STEERING_RATE, MAX_ANGLE)
        else:
            # Return to center
            if angle > 0:
                angle = max(0, angle - RETURN_RATE)
            elif angle < 0:
                angle = min(0, angle + RETURN_RATE)
                
        return False, angle, is_stopped, current_speed
        
    except Exception as e:
        print(f"Error in handle_input: {str(e)}")
        return False, angle, is_stopped, current_speed

def draw_controls(screen, angle, is_stopped, keys):
    """Draw control indicators"""
    screen.fill((0, 0, 0))
    
    # Draw stop indicator
    if is_stopped:
        pygame.draw.circle(screen, (255, 0, 0), (160, 120), 20)
        return
        
    # Draw steering indicators
    if keys[pygame.K_LEFT]:
        pygame.draw.circle(screen, (255, 0, 0), (40, 200), 10)
    if keys[pygame.K_RIGHT]:
        pygame.draw.circle(screen, (255, 0, 0), (280, 200), 10)
    if keys[pygame.K_UP]:
        pygame.draw.circle(screen, (255, 0, 0), (160, 20), 10)    
        
    # Draw steering position
    indicator_x = max(0, min(310, 160 + (angle * 320)))
    pygame.draw.rect(screen, (255, 255, 255), (indicator_x, 120, 10, 10))

def draw_debug_info(screen, font, clock):
    """Draw minimal debug info"""
    fps = clock.get_fps()
    text = font.render(f"FPS: {fps:.1f}", True, (255, 255, 255))
    screen.blit(text, (10, 10))

def robot_control_thread(bot):
    """Handle robot commands in separate thread"""
    while True:
        try:
            while not command_queue.empty():
                command = command_queue.get_nowait()
                if command is None:
                    return
                left, right = command
                bot.setVelocity(left, right)
        except Queue.Empty:
            pass
        time.sleep(1/CONTROL_FPS)

def image_capture_thread(bot, save_dir, im_num):
    """Handle image capture in separate thread"""
    local_im_num = im_num
    last_capture = time.time()
    
    while True:
        if time.time() - last_capture >= 1/CAPTURE_FPS:
            command = image_queue.get()
            if command is None:
                break
                
            angle, left, right = command
            img = bot.getImage()
            if img is not None:
                image_name = f"{str(local_im_num).zfill(6)}_{angle:.2f}_{left}_{right}.jpg"
                cv2.imwrite(os.path.join(save_dir, image_name), img)
                local_im_num += 1
                last_capture = time.time()
        time.sleep(0.001)  # Small sleep to prevent busy waiting

class DummyBot:
    """Mock robot for testing"""
    def __init__(self, ip=None):
        self.left = 0
        self.right = 0
        self.min_latency = 0.01  # 10ms minimum
        self.max_latency = 0.05  # 50ms maximum
        
    def setVelocity(self, left, right):
        # Simulate network latency
        time.sleep(random.uniform(self.min_latency, self.max_latency))
        self.left = left
        self.right = right
        
    def getImage(self):
        # Return test pattern instead of black image
        img = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.putText(img, "TEST MODE", (80, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return img

def get_bot(test_mode=False, ip='192.168.1.179'):
    """Get real or dummy bot based on mode"""
    if test_mode:
        return DummyBot()
    elif PiBot is not None:
        return PiBot(ip)
    return DummyBot()  # Fallback if PiBot unavailable

def main():
    pygame.init()
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((320, 240))
    font = pygame.font.Font(None, 24)
    bot = get_bot(args.test, args.ip)

    # Initialize threads for both real and test modes
    control_thread = threading.Thread(target=robot_control_thread, args=(bot,), daemon=True)
    control_thread.start()

    if args.collect:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = os.path.join(script_path, '..', 'data', args.folder, timestamp)
        os.makedirs(save_dir, exist_ok=True)
        image_thread = threading.Thread(target=image_capture_thread, 
                                     args=(bot, save_dir, args.im_num),
                                     daemon=True)
        image_thread.start()
    
    angle = 0
    is_stopped = False
    current_speed = BASE_SPEED
    
    try:
        while True:
            should_quit, angle, is_stopped, current_speed = handle_input(angle, is_stopped, current_speed)
            if should_quit:
                if not args.test:
                    command_queue.put(None)  # Signal threads to stop
                    if args.collect:
                        image_queue.put(None)
                pygame.quit()
                return

            if not args.test and not is_stopped:
                # Clear queue before adding new command
                while not command_queue.empty():
                    try:
                        command_queue.get_nowait()
                    except Queue.Empty:
                        break
                        
                left = int(current_speed + TURN_SPEED * angle)
                right = int(current_speed - TURN_SPEED * angle)
                command_queue.put((left, right))
                if args.collect:
                    image_queue.put((angle, left, right))  # Send tuple with all values

            screen.fill((0, 0, 0))
            draw_controls(screen, angle, is_stopped, pygame.key.get_pressed())
            draw_debug_info(screen, font, clock)
            pygame.display.flip()
            clock.tick(CONTROL_FPS)

    except KeyboardInterrupt:
        if not args.test:
            command_queue.put(None)
            if args.collect:
                image_queue.put(None)
        pygame.quit()

if __name__ == "__main__":
    main()