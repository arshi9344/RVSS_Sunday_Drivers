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
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        
    def __len__(self):        
        return len(self.filenames)
    
    def __getitem__(self,idx):
        f = self.filenames[idx]        
        img = cv2.imread(f)[120:, :, :]
        
        # Apply CLAHE to each channel
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        img_lab[:,:,0] = self.clahe.apply(img_lab[:,:,0])
        img = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)
        
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
            
        return img, speeds
