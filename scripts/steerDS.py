import numpy as np
from glob import glob
from torchvision import transforms
from torch.utils.data import Dataset
import cv2
from glob import glob
from os import path
import torch

class SteerDataSet(Dataset):
    
    def __init__(self,root_folder,img_ext = ".jpg" , transform=None):
        self.root_folder = root_folder
        self.transform = transform        
        self.img_ext = img_ext        
        self.filenames = glob(path.join(self.root_folder,"*" + self.img_ext))            
        self.totensor = transforms.ToTensor()
        self.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
        
    def __len__(self):        
        return len(self.filenames)
    
    def __getitem__(self,idx):
        try:
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

            # Grayscale version
            # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # img = self.clahe.apply(img_gray)
            # img = np.expand_dims(img, axis=2)  # Add channel dimension back

            if self.transform is None:
                img = self.totensor(img)
            else:
                img = self.transform(img)   
            
            # Parse filename for wheel speeds
            filename = path.basename(f)
            parts = filename.split('_')
            speeds = torch.tensor([
                float(parts[-2]),
                float(parts[-1].split('.jpg')[0])
            ])
            
            # Normalize to [0,1] using min-max normalization
            speeds = (speeds) / (60.0)  # Given speed range 0-60
            speeds = torch.clamp(speeds, 0, 1)
            # print(f"Loading {f}, shape: {img.shape if img is not None else 'None'}")    
            return img, speeds
        except Exception as e:
            print(f"Error loading index {idx}: {str(e)}")
            raise
