#!/usr/bin/env python
import sys
import os
import cv2
import pygame
import argparse
from datetime import datetime

os.environ['SDL_AUDIODRIVER'] = 'dummy'
# os.environ['SDL_VIDEODRIVER'] = 'dummy'  # Comment this out

script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_path, "../PenguinPi-robot/software/python/client/")))
from pibot_client import PiBot

# Parse arguments
parser = argparse.ArgumentParser(description='PiBot client')
parser.add_argument('--ip', type=str, default='localhost', help='IP address of PiBot')
parser.add_argument('--im_num', type=int, default=0)
parser.add_argument('--folder', type=str, default='train')
args = parser.parse_args()

# Constants
MAX_ANGLE = 0.5
STEERING_RATE = 0.3
BASE_SPEED = 60
TURN_SPEED = 40
FPS = 30
collect_data = bool(0)

KEY_REPEAT_DELAY = 100  # ms before key repeat
KEY_REPEAT_INTERVAL = 50  # ms between repeats
RETURN_RATE = 0.1  # Rate of return to center

def update_steering(keys, angle, delta_time):
    """Update steering angle based on key states"""
    if keys[pygame.K_LEFT]:
        return max(angle - STEERING_RATE * delta_time, -MAX_ANGLE)
    elif keys[pygame.K_RIGHT]:
        return min(angle + STEERING_RATE * delta_time, MAX_ANGLE)
    else:
        # Return to center
        if angle > 0:
            return max(0, angle - RETURN_RATE * delta_time)
        elif angle < 0:
            return min(0, angle + RETURN_RATE * delta_time)
    return angle

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
        
    # Draw steering position
    indicator_x = max(0, min(310, 160 + (angle * 320)))
    pygame.draw.rect(screen, (255, 255, 255), (indicator_x, 120, 10, 10))

def main():
    # Create timestamped data folder
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(script_path, '..', 'data', args.folder, timestamp)
    os.makedirs(save_dir, exist_ok=True)

    pygame.init()
    pygame.key.set_repeat(KEY_REPEAT_DELAY, KEY_REPEAT_INTERVAL)
    screen = pygame.display.set_mode((320, 240))
    pygame.display.set_caption('Robot Control')
    clock = pygame.time.Clock()
    
    bot = PiBot(ip=args.ip)
    angle = 0
    is_stopped = False
    
    # Track key states
    key_states = {
        pygame.K_LEFT: False,
        pygame.K_RIGHT: False,
        pygame.K_SPACE: False
    }
    last_update = pygame.time.get_ticks()
    
    try:
        while True:
            current_time = pygame.time.get_ticks()
            delta_time = (current_time - last_update) / 1000.0  # Convert to seconds
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and 
                    event.key == pygame.K_q):
                    return
                # Toggle stop state on KEYDOWN only
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    is_stopped = not is_stopped
                    if is_stopped:
                        bot.setVelocity(0, 0)
                if event.type == pygame.KEYDOWN:
                    if event.key in key_states:
                        key_states[event.key] = True
                elif event.type == pygame.KEYUP:
                    if event.key in key_states:
                        key_states[event.key] = False
            
            screen.fill((0, 0, 0))
            
            if is_stopped:
                pygame.draw.circle(screen, (255, 0, 0), (160, 120), 20)
            else:
                # Normal operation code here
                keys = pygame.key.get_pressed()
                # Use smaller steering rate for smoother control
                STEERING_RATE = 0.05
                
                # Immediate response to key states
                if keys[pygame.K_LEFT]:
                    angle = max(angle - STEERING_RATE, -MAX_ANGLE)
                    pygame.draw.circle(screen, (255, 0, 0), (40, 200), 10)
                elif keys[pygame.K_RIGHT]:
                    angle = min(angle + STEERING_RATE, MAX_ANGLE)
                    pygame.draw.circle(screen, (255, 0, 0), (280, 200), 10)
                else:
                    # Faster return to center
                    if angle > 0:
                        angle = max(0, angle - STEERING_RATE * 2)
                    elif angle < 0:
                        angle = min(0, angle + STEERING_RATE * 2)
                
                # Draw steering indicator
                pygame.draw.rect(screen, (255, 255, 255), 
                               (160 + (angle * 160), 120, 10, 10))
                
                # Update robot
                left = int(BASE_SPEED + TURN_SPEED * angle)
                right = int(BASE_SPEED - TURN_SPEED * angle)
                bot.setVelocity(left, right)
                
                # Save image
                if collect_data == True:    #collect data flag
                    img = bot.getImage()
                    if img is not None:
                        image_path = os.path.join(save_dir, f"{str(args.im_num).zfill(6)}_{angle:.2f}.jpg")
                        cv2.imwrite(image_path, img)
                    args.im_num += 1
            
            # Update steering with time-based movement
            if key_states[pygame.K_LEFT]:
                angle = max(angle - STEERING_RATE * delta_time, -MAX_ANGLE)
            elif key_states[pygame.K_RIGHT]:
                angle = min(angle + STEERING_RATE * delta_time, MAX_ANGLE)
            else:
                # Return to center based on time
                if angle > 0:
                    angle = max(0, angle - RETURN_RATE * delta_time)
                elif angle < 0:
                    angle = min(0, angle + RETURN_RATE * delta_time)
            
            last_update = current_time
            
            pygame.display.flip()
            clock.tick(FPS)
            
    except KeyboardInterrupt:
        pass
    finally:
        bot.setVelocity(0, 0)
        pygame.quit()

if __name__ == '__main__':
    main()