import threading
from queue import Queue, Empty, Full
import time
import os
import cv2
import sys
import numpy as np

script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_path, "../PenguinPi-robot/software/python/client/")))

try:
    from pibot_client import PiBot 
    PiBot = PiBot
except ImportError:
    PiBot = None

COMMAND_RATE = 10  # Unified command rate at 20Hz
CAPTURE_FPS = 30   

class RobotController:
    def __init__(self, bot, debug=False):
        self.bot = bot
        self.debug = debug
        self.command_rate = COMMAND_RATE
        self.capture_fps = CAPTURE_FPS
        self.command_queue = Queue(maxsize=1)  # Increased for smoother control
        self.image_queue = Queue(maxsize=10)  # For deployment
        self.save_queue = Queue(maxsize=10)   # For saving to disk
        self._running = False
        self._threads = {}
        self._last_command = None

    def _control_thread(self):
        """Handle robot control commands at fixed rate"""
        self._threads['control'] = {'active': True, 'healthy': True}
        interval = 1.0 / self.command_rate
        last_time = time.time()
        
        while self._running and self._threads['control']['active']:
            try:
                current_time = time.time()
                if current_time - last_time >= interval:
                    try:
                        command = self.command_queue.get_nowait()
                        if command is None:
                            break
                        left, right = command
                        self.bot.setVelocity(left, right, None, 0.5)  # Keep original method name
                        if self.debug:
                            print(f"[DEBUG] Sent command: L={left} R={right}")
                    except Empty:
                        pass
                    last_time = current_time
                time.sleep(0.001)
            except Exception as e:
                self._threads['control']['healthy'] = False
                if self.debug:
                    print(f"[ERROR] Control thread error: {e}")
                break

    def _image_thread(self, save_dir=None, im_num=0):
        """Handle image capture and return/save"""
        self._threads['image'] = {'active': True, 'healthy': True}
        last_capture = time.time()
        local_im_num = im_num

        while self._running and self._threads['image']['active']:
            try:
                if time.time() - last_capture >= 1/self.capture_fps:
                    img = self.bot.getImage()
                    if img is not None:
                        # Put image in queue for retrieval
                        try:
                            self.image_queue.put_nowait(img)
                        except Full:
                            # Clear old image if queue full
                            try:
                                self.image_queue.get_nowait()
                                self.image_queue.put_nowait(img)
                            except Empty:
                                pass
                                
                        # Save image if directory provided
                        if save_dir:
                            try:
                                command = self.command_queue.get_nowait()
                                angle, left, right, is_stopped = command
                                if not is_stopped:
                                    image_name = f"{str(local_im_num).zfill(6)}_{angle:.2f}_{left}_{right}.jpg"
                                    cv2.imwrite(os.path.join(save_dir, image_name), img)
                                    local_im_num += 1
                            except Empty:
                                pass
                                
                    last_capture = time.time()
                    
                time.sleep(0.001)
            except Exception as e:
                self._threads['image']['healthy'] = False
                if self.debug:
                    print(f"[ERROR] Image thread error: {e}")
                break

    def start(self, save_dir=None, im_num=0):
        """Start robot control and optional image capture threads"""
        if self._running:
            return False
            
        self._running = True
        
        # Start control thread
        self.control_thread = threading.Thread(target=self._control_thread, daemon=True)
        self.control_thread.start()
        
        # Start image thread if needed
        if save_dir:
            self.image_thread = threading.Thread(
                target=self._image_thread,
                args=(save_dir, im_num),
                daemon=True
            )
            self.image_thread.start()
            
        # Verify threads started
        time.sleep(0.1)  # Brief pause to let threads initialize
        return all(t['healthy'] for t in self._threads.values())

    def stop(self):
        """Stop all threads and cleanup"""
        if not self._running:
            return
            
        self._running = False
        
        # Signal threads to stop
        for thread_info in self._threads.values():
            thread_info['active'] = False
            
        # Clear queues
        self.command_queue.put(None)
        if 'image' in self._threads:
            self.image_queue.put(None)
            
        # Stop robot through control thread
        self.queue_command(0, 0)
        
        # Wait for threads to finish
        if hasattr(self, 'control_thread'):
            self.control_thread.join(timeout=1.0)
        if hasattr(self, 'image_thread'):
            self.image_thread.join(timeout=1.0)
            
        self._threads.clear()

    def queue_command(self, left, right):
        """Queue velocity command for control thread"""
        try:
            # Clear existing command
            while not self.command_queue.empty():
                self.command_queue.get_nowait()
            # Add new command
            self.command_queue.put_nowait((left, right))
            if self.debug:
                print(f"[DEBUG] Queued command: L={left} R={right}")
        except Full:
            if self.debug:
                print("[DEBUG] Command queue full - skipping")

def get_bot(test_mode=False, ip='localhost', debug=False):
    """Initialize robot connection with timeout
    
    Args:
        test_mode (bool): Use dummy bot if True
        ip (str): Robot IP address
        debug (bool): Enable debug logging
    
    Returns:
        Union[PiBot, DummyBot]: Connected robot instance or dummy
    """
    def debug_print(msg):
        if debug:
            print(f"[DEBUG] {msg}")

    if test_mode or PiBot is None:
        debug_print("Using dummy bot")
        return DummyBot()

    connection_result = Queue()
    
    def connect_with_timeout():
        try:
            bot = PiBot(ip=ip)
            bot.setVelocity(0, 0)
            connection_result.put(('success', bot))
        except Exception as e:
            connection_result.put(('error', str(e)))
    
    # Start connection thread
    debug_print(f"Connecting to robot at {ip}")
    connect_thread = threading.Thread(target=connect_with_timeout, daemon=True)
    connect_thread.start()
    
    try:
        # Wait for connection with timeout
        result = connection_result.get(timeout=5.0)
        status, data = result
        
        if status == 'success':
            debug_print("Robot connection verified")
            return data
        else:
            debug_print(f"Connection failed: {data}")
            return DummyBot()
            
    except Empty:
        debug_print("Connection timeout")
        return DummyBot()
    except Exception as e:
        debug_print(f"Unexpected error: {e}")
        return DummyBot()

class DummyBot:
    """Mock robot for testing"""
    def __init__(self, ip=None):
        self.left = 0
        self.right = 0
        self._test_image = None  # Cache for test image
        
    def setVelocity(self, left, right):
        """Set the velocity for left and right wheels.
        
        Args:
            left (int): Left wheel velocity (-100 to 100)
            right (int): Right wheel velocity (-100 to 100)
        """
        self.left = max(-100, min(100, int(left)))
        self.right = max(-100, min(100, int(right)))
        
    def getImage(self):
        """Get a test image with text overlay.
        
        Returns:
            numpy.ndarray: 320x240 RGB test image
        """
        if self._test_image is None:
            self._test_image = np.zeros((240, 320, 3), dtype=np.uint8)
            cv2.putText(self._test_image, "TEST MODE", (80, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return self._test_image.copy()

__all__ = ['RobotController', 'get_bot']
