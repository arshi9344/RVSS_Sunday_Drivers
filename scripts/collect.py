#!/usr/bin/env python
import os
import pygame
import argparse
import threading
import time
from datetime import datetime
from queue import Queue, Empty
from robot_comms import RobotController, get_bot

os.environ['SDL_AUDIODRIVER'] = 'dummy'
# os.environ['SDL_VIDEODRIVER'] = 'dummy'  # Comment this out
script_path = os.path.dirname(os.path.realpath(__file__))

# Parse arguments
parser = argparse.ArgumentParser(description='PiBot client')
parser.add_argument('--ip', type=str, default='192.168.1.100', 
                   help='IP address of PiBot (default: 192.168.1.100)')
parser.add_argument('--im_num', type=int, default=0)
parser.add_argument('--folder', type=str, default='train')
parser.add_argument('--collect', action='store_true', help='Enable data collection')
parser.add_argument('--test', action='store_true', default=False, help='Run in test mode without robot')
parser.add_argument('--debug', action='store_true', help='Enable debug mode')
args = parser.parse_args()



# Constants
FPS = 60  # Display update rate
COMMAND_RATE = 20  # Match robot_comms.py rate
MAX_ANGLE = 0.5
BASE_SPEED = 40
TURN_SPEED = 40
RETURN_RATE = 0.08
STEERING_RATE = 0.05
MAX_BOOST_SPEED = 60
BOOST_RATE = 0.2
ANGLE_THRESHOLD = 0.1

# Thread synchronization
stop_event = threading.Event()

def handle_input(angle, is_stopped, current_speed=BASE_SPEED):
    """Process input and return updated control state"""
    keys = pygame.key.get_pressed()
    left = right = current_speed
    
    # Check quit/space events (these can't be missed)
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (
            event.type == pygame.KEYDOWN and 
            event.key == pygame.K_q):
            return True, angle, is_stopped, current_speed, (0, 0)
        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            is_stopped = not is_stopped
            return False, angle, is_stopped, current_speed, (0, 0)

    # Process continuous inputs
    if not is_stopped:
        # Speed control
        if keys[pygame.K_UP]:
            current_speed = min(current_speed + BOOST_RATE, MAX_BOOST_SPEED)
        else:
            current_speed = max(current_speed - BOOST_RATE, BASE_SPEED)
            
        # Steering control
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
                
        # Calculate motor speeds
        left = round(current_speed + TURN_SPEED * angle)
        right = round(current_speed - TURN_SPEED * angle)
        
    return False, angle, is_stopped, current_speed, (left, right)

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

def draw_debug_info(screen, font, clock, left=None, right=None):
    """Draw FPS and wheel speeds"""
    y = 10
    
    # Draw FPS
    fps = clock.get_fps()
    fps_text = font.render(f"FPS: {fps:.1f}", True, (255, 255, 255))
    screen.blit(fps_text, (10, y))
    
    # Draw wheel speeds if available
    if left is not None and right is not None:
        y += 20
        speeds_text = font.render(f"L: {left} R: {right}", True, (255, 255, 255))
        screen.blit(speeds_text, (10, y))

def debug_print(msg, debug=False):
    if debug:
        print(f"[DEBUG] {msg}")

def command_sender(robot, debug=False):
    """Thread to handle sending commands at fixed rate"""
    interval = 10 #1.0 / COMMAND_RATE
    last_time = time.time()
    
    while not stop_event.is_set():
        try:
            if time.time() - last_time >= interval:
                try:
                    command = robot.command_queue.get_nowait()
                    robot.setVelocity(*command)
                    debug_print(f"Sent command: {command}", debug)
                except Empty:
                    pass
                last_time = time.time()
            time.sleep(0.001)  # Prevent CPU spinning
        except Exception as e:
            debug_print(f"Command sender error: {e}", debug)

# Verify test mode setting
debug_print(f"Args parsed: {vars(args)}", args.debug)

def main():
    pygame.init()
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((320, 240))
    font = pygame.font.Font(None, 24)
    
    try:

        # Verify args before passing
        debug_print(f"Initializing with test={args.test}, ip={args.ip}", args.debug)
        bot = get_bot(test_mode=args.test, ip=args.ip)  # Use explicit keyword args
        robot = RobotController(bot, debug=args.debug)
        
        if args.collect:
            save_dir = os.path.join(script_path, '..', 'data', args.folder,
                                  datetime.now().strftime('%Y%m%d_%H%M%S'))
            os.makedirs(save_dir, exist_ok=True)
            robot.start(save_dir, args.im_num)
        else:
            robot.start()
        angle = 0
        is_stopped = False
        current_speed = BASE_SPEED
        should_quit = False
        last_speeds = (0, 0)
        last_stop_state = False

        while True:
            # Get latest state
            should_quit, angle, is_stopped, current_speed, speeds = \
                handle_input(angle, is_stopped, current_speed)

            if should_quit:
                robot.queue_command(0, 0)
                robot.stop()
                break

            # Handle stop state changes only
            if is_stopped != last_stop_state:
                robot.queue_command(0, 0)
                last_speeds = (0, 0)
                last_stop_state = is_stopped
                if args.debug:
                    debug_print(f"Stop state changed: {is_stopped}")

            # Only process movement commands when not stopped
            if not is_stopped:
                if speeds != last_speeds:
                    robot.queue_command(*speeds)
                    last_speeds = speeds
            
            # Update display
            screen.fill((0, 0, 0))
            draw_controls(screen, angle, is_stopped, pygame.key.get_pressed())
            draw_debug_info(screen, font, clock, *speeds)
            pygame.display.flip()
            clock.tick(FPS)
            #print(*speeds)

            if args.collect:
                robot.save_queue.put((angle, *speeds, is_stopped))

    except KeyboardInterrupt:
        debug_print("Keyboard interrupt detected", args.debug)
    finally:
        if 'robot' in locals():
            robot.queue_command(0, 0)  # Ensure motors stop
            robot.stop()  # Ensure threads stop
        pygame.quit()

if __name__ == "__main__":
    main()