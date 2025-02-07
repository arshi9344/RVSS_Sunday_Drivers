import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import os
from pathlib import Path
import glob

class ImageViewer:
    def __init__(self, root, folder):
        self.root = root
        self.root.title("Image Viewer")
        
        # Image loading
        self.folder = folder
        self.image_paths = []
        self.current_index = 0
        self.selected_images = set()
        self.slideshow_running = False
        self.slideshow_reverse_running = False
        self.slideshow_after_id = None
        
        # UI Setup
        self.setup_ui()
        
        # Key bindings
        self.root.bind('<Left>', lambda e: self.show_previous())
        self.root.bind('<Right>', lambda e: self.show_next())
        self.root.bind('<space>', lambda e: self.toggle_selection())
        self.root.bind('<Delete>', lambda e: self.delete_selected())
        
    def setup_ui(self):
        # Main image display
        self.image_label = tk.Label(self.root)
        self.image_label.pack(pady=10)
        
        # Label to display tags from filename (last three underscore separated fields)
        self.tags_var = tk.StringVar()
        self.tags_label = ttk.Label(self.root, textvariable=self.tags_var, font=("Arial", 12))
        self.tags_label.pack(pady=5)
        
        # Navigation and selection controls
        nav_controls = ttk.Frame(self.root)
        nav_controls.pack(pady=5)
        ttk.Button(nav_controls, text="Previous", command=self.show_previous).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_controls, text="Next", command=self.show_next).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_controls, text="Toggle Select", command=self.toggle_selection).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_controls, text="Delete Selected", command=self.delete_selected).pack(side=tk.LEFT, padx=5)
        # New button to move to the next folder in the data directory
        ttk.Button(nav_controls, text="Next Folder", command=self.move_to_next_folder).pack(side=tk.LEFT, padx=5)
        
        # Slideshow controls
        slide_controls = ttk.Frame(self.root)
        slide_controls.pack(pady=5)
        ttk.Button(slide_controls, text="Start Slideshow", command=self.start_slideshow).pack(side=tk.LEFT, padx=5)
        ttk.Button(slide_controls, text="Start Reverse Slideshow", command=self.start_reverse_slideshow).pack(side=tk.LEFT, padx=5)
        ttk.Button(slide_controls, text="Stop Slideshow", command=self.stop_slideshow).pack(side=tk.LEFT, padx=5)
        
        # New button to mark the current folder as processed
        process_controls = ttk.Frame(self.root)
        process_controls.pack(pady=5)
        ttk.Button(process_controls, text="Mark Folder Processed", command=self.mark_folder_processed).pack(side=tk.LEFT, padx=5)
        
        # Range selection controls using sliders
        range_frame = ttk.LabelFrame(self.root, text="Select Range for Deletion")
        range_frame.pack(pady=10, fill="x", padx=10)
        
        # Add mark buttons next to sliders
        slider_frame = ttk.Frame(range_frame)
        slider_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        
        ttk.Label(slider_frame, text="Start Index:").pack(side=tk.LEFT)
        self.start_scale = tk.Scale(slider_frame, from_=0, to=0, orient=tk.HORIZONTAL)
        self.start_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(slider_frame, text="Mark Start", 
                   command=lambda: self.start_scale.set(self.current_index)).pack(side=tk.LEFT)
        
        slider_frame2 = ttk.Frame(range_frame)
        slider_frame2.grid(row=1, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        
        ttk.Label(slider_frame2, text="End Index:").pack(side=tk.LEFT)
        self.end_scale = tk.Scale(slider_frame2, from_=0, to=0, orient=tk.HORIZONTAL)
        self.end_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(slider_frame2, text="Mark End",
                   command=lambda: self.end_scale.set(self.current_index)).pack(side=tk.LEFT)
        
        # Button controls with prominent delete button
        button_frame = ttk.Frame(range_frame)
        button_frame.grid(row=2, column=0, columnspan=2, pady=10)
        
        ttk.Button(button_frame, text="Select Range", 
                   command=self.select_range).pack(side=tk.LEFT, padx=5)
        
        delete_button = tk.Button(button_frame, text="DELETE SELECTED", 
                                command=self.delete_selected,
                                background="red", foreground="white",
                                font=("Arial", 10, "bold"))
        delete_button.pack(side=tk.LEFT, padx=5)
        
        range_frame.columnconfigure(1, weight=1)
        
        # Status bar
        self.status_var = tk.StringVar()
        ttk.Label(self.root, textvariable=self.status_var).pack(pady=5)
        
    def load_images(self):
        """Load all images from the specified folder"""
        extensions = ('*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp')
        self.image_paths = []
        for ext in extensions:
            self.image_paths.extend(glob.glob(os.path.join(self.folder, ext)))
        self.image_paths.sort()
        self.current_index = 0
        
        # Update slider ranges based on number of images loaded
        num_images = len(self.image_paths)
        if num_images > 0:
            self.start_scale.config(from_=0, to=num_images-1)
            self.end_scale.config(from_=0, to=num_images-1)
            self.start_scale.set(0)
            self.end_scale.set(num_images-1)
        
        self.show_current_image()
        
    def show_current_image(self):
        if not self.image_paths:
            return
        
        try:
            image = Image.open(self.image_paths[self.current_index])
        except Exception as e:
            print(f"Error loading {self.image_paths[self.current_index]}: {e}")
            return
        
        display_size = (800, 600)
        image.thumbnail(display_size, Image.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=photo)
        self.image_label.image = photo
        
        # Update tags label from filename (last three underscore-separated fields)
        filename = os.path.basename(self.image_paths[self.current_index])
        parts = filename.split("_")
        if len(parts) >= 3:
            tags = parts[-3:]
            tags[-1] = tags[-1].split(".")[0]
            tag_text = " | ".join(tags)
        else:
            tag_text = "No tags available"
        self.tags_var.set(tag_text)
        
        status = f"Image {self.current_index + 1}/{len(self.image_paths)}"
        if self.image_paths[self.current_index] in self.selected_images:
            status += " (Selected)"
        self.status_var.set(status)
        
    def show_next(self):
        if self.current_index < len(self.image_paths) - 1:
            self.current_index += 1
            self.show_current_image()
            
    def show_previous(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.show_current_image()
            
    def toggle_selection(self):
        current_path = self.image_paths[self.current_index]
        if current_path in self.selected_images:
            self.selected_images.remove(current_path)
        else:
            self.selected_images.add(current_path)
        self.show_current_image()
        
    def select_range(self):
        """Selects all images in the range defined by the two sliders."""
        # Clear previous selection
        self.selected_images.clear()
        
        start_index = int(self.start_scale.get())
        end_index = int(self.end_scale.get())
        if start_index > end_index:
            start_index, end_index = end_index, start_index
            
        for i in range(start_index, end_index + 1):
            self.selected_images.add(self.image_paths[i])
        self.show_current_image()
        
    def delete_selected(self):
        if not self.selected_images:
            return
            
        for path in list(self.selected_images):
            try:
                os.remove(path)
                self.image_paths.remove(path)
                self.selected_images.discard(path)
            except OSError as e:
                print(f"Error deleting {path}: {e}")
                
        if self.image_paths:
            start_index = int(self.start_scale.get())
            self.current_index = start_index - 1 if start_index > 0 else 0

            num_images = len(self.image_paths)
            self.start_scale.config(to=num_images - 1)
            self.end_scale.config(to=num_images - 1)
            self.show_current_image()
        else:
            self.image_label.configure(image='')
            self.tags_var.set("")
            self.status_var.set("No images remaining")
            
    def start_slideshow(self):
        """Start auto-playing images forward at 30fps."""
        self.slideshow_running = True
        self.slideshow_reverse_running = False
        self.slideshow_step()
        
    def start_reverse_slideshow(self):
        """Start auto-playing images in reverse at 30fps."""
        self.slideshow_reverse_running = True
        self.slideshow_running = False
        self.slideshow_reverse_step()
        
    def stop_slideshow(self):
        """Stop any auto-playing slideshow."""
        self.slideshow_running = False
        self.slideshow_reverse_running = False
        if self.slideshow_after_id is not None:
            self.root.after_cancel(self.slideshow_after_id)
            self.slideshow_after_id = None
            
    def slideshow_step(self):
        if not self.slideshow_running or not self.image_paths:
            return
        self.show_next()
        if self.current_index == len(self.image_paths) - 1:
            self.current_index = -1
        self.slideshow_after_id = self.root.after(33, self.slideshow_step)
        
    def slideshow_reverse_step(self):
        if not self.slideshow_reverse_running or not self.image_paths:
            return
        self.show_previous()
        if self.current_index <= 0:
            self.current_index = len(self.image_paths)
        self.slideshow_after_id = self.root.after(33, self.slideshow_reverse_step)
        
    def mark_folder_processed(self):
        """Mark the current folder as processed by appending it to a tracking file."""
        record_file = os.path.join(os.path.dirname(__file__), "processed_folders.txt")
        folder_path = os.path.abspath(self.folder)
        processed = []
        if os.path.exists(record_file):
            with open(record_file, "r") as f:
                processed = [line.strip() for line in f.readlines()]
        if folder_path not in processed:
            with open(record_file, "a") as f:
                f.write(folder_path + "\n")
            self.status_var.set(f"Folder marked as processed: {folder_path}")
        else:
            self.status_var.set(f"Folder already processed: {folder_path}")

    def move_to_next_folder(self):
        """
        Change self.folder to the next folder (alphabetically) in the parent directory
        and reload images.
        """
        parent_dir = os.path.dirname(os.path.abspath(self.folder))
        # Get all subdirectories (full paths) in the parent directory
        folders = sorted([os.path.join(parent_dir, d) for d in os.listdir(parent_dir)
                          if os.path.isdir(os.path.join(parent_dir, d))])
        current_folder = os.path.abspath(self.folder)
        if current_folder in folders:
            current_index = folders.index(current_folder)
            next_index = current_index + 1
            if next_index < len(folders):
                self.folder = folders[next_index]
                self.load_images()
                self.status_var.set(f"Moved to folder: {self.folder}")
            else:
                self.status_var.set("No more folders in the parent directory.")
        else:
            self.status_var.set("Current folder not found among siblings.")

if __name__ == "__main__":
    root = tk.Tk()
    folder = filedialog.askdirectory(title="Select Image Folder")
    if not folder:
        print("No folder selected. Exiting.")
        root.destroy()
    else:
        viewer = ImageViewer(root, folder)
        viewer.load_images()
        root.mainloop()