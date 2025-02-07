import numpy as np
from glob import glob
from torchvision import transforms
from torch.utils.data import Dataset
import cv2
from glob import glob
from os import path
import torch
from PIL import Image, ImageOps

class SteerDataSet(Dataset):
    
    def __init__(self,root_folder,img_ext = ".jpg" , transform=None, double_with_flip=True):
        self.root_folder = root_folder
        self.transform = transform        
        self.img_ext = img_ext        
        self.filenames = glob(path.join(self.root_folder,"*" + self.img_ext))            
        self.totensor = transforms.ToTensor()
        self.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
        self.double_with_flip = double_with_flip
        
    def __len__(self):        
        if self.double_with_flip:
            return len(self.filenames) * 2  # original + flipped
        return len(self.filenames)
    
    def __getitem__(self,idx):
        try:
            # If idx < length, use the original; otherwise, flip.
            is_flipped = False
            if self.double_with_flip and idx >= len(self.filenames):
                idx = idx - len(self.filenames)
                is_flipped = True

            f = self.filenames[idx]        
            img = cv2.imread(f)[120:, :, :]  # Gets bottom half of image
            
            if img is None:
                raise ValueError(f"Failed to load image: {f}")

            # RGB version     
            # # Convert BGR to HSV
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img_hsv[:,:,2] = self.clahe.apply(img_hsv[:,:,2])  # Apply CLAHE to V channel
            # Convert HSV to RGB
            img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)

            # Parse filename for wheel speeds
            filename = path.basename(f)
            parts = filename.split('_')
            speeds = torch.tensor([
                float(parts[-2]),
                float(parts[-1].split('.jpg')[0])
            ])

            if is_flipped:
                #print(f"flipped.")
                # Flip horizontally
                img = np.ascontiguousarray(np.flip(img, axis=1))
                # Swap left/right speeds
                speeds[0], speeds[1] = speeds[1], speeds[0]

            # Convert to PIL for torchvision transforms
            pil_img = Image.fromarray(img)

            if self.transform is not None:
                pil_img = self.transform(pil_img)

            # Normalize to [0,1] using min-max normalization
            speeds = (speeds) / (60.0)  # Given speed range 0-60
            speeds = torch.clamp(speeds, 0, 1)

            # Return both image and speeds
            return pil_img, speeds
        except Exception as e:
            print(f"Error loading index {idx}: {str(e)}")
            raise
