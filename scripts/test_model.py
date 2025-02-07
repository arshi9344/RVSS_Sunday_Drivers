import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import os
import sys

# Define the same network architecture as in deploy.py
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # 1) Convolution + Pooling block
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        
        # 2) Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2)

        # 3) Fully connected
        # The in_features must match the flattened size after conv/pool operations.
        # Adjust this if your network differs.
        self.fc1 = nn.Linear(10816, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 2)  # final 2 outputs (e.g. left/right speeds)

        self.relu = nn.ReLU()

    def forward(self, x):
        # Input shape: [batch_size, 3, 60, 60]
        x = self.pool(self.relu(self.conv1(x)))  # [batch_size, 16, ? , ?]
        x = self.pool(self.relu(self.conv2(x)))  # [batch_size, 32, ? , ?]
        x = self.pool(self.relu(self.conv3(x)))  # [batch_size, 64, ? , ?]
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        out = self.fc3(x)
        return out

def load_model(model_path, device):
    model = Net().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def load_and_preprocess_image(image_path, target_size=(120, 120)):
    # Load image with PIL 
    img = Image.open(image_path).convert('RGB')
    # Use the same transforms as deploy.py
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(target_size),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    img_tensor = preprocess(img).unsqueeze(0)  # Add batch dimension
    return img_tensor

def extract_ground_truth(filename):
    """Extract actual speeds from filename like 'image_[left]_[right].jpg'"""
    parts = filename.split('_')
    if len(parts) >= 3:
        try:
            left = float(parts[-2])
            right = float(parts[-1].split('.')[0])
            return np.array([left, right])
        except ValueError:
            return None
    return None

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Test model in PyTorch')
    parser.add_argument('--model_path', type=str, 
                       default=os.path.join(os.path.dirname(__file__), "ADAM_Models", "best_model_10.pth"), 
                       help='Path to the model weights')
    parser.add_argument('--folder_path', type=str, required=True, 
                       help='Path to folder containing test images')
    parser.add_argument('--device', type=str, default='cpu', 
                       help='Device to run the model (cpu or cuda)')
    args = parser.parse_args()

    # Ensure paths are absolute
    model_path = os.path.abspath(args.model_path)
    folder_path = os.path.abspath(args.folder_path)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Image folder not found at: {folder_path}")

    device = torch.device(args.device)

    print(f"Looking for model at: {model_path}")
    print(f"Looking for images in: {folder_path}")
    print(f"Using device: {device}")

    model = load_model(model_path, device)
    print("Model loaded successfully")

    # Get list of image files first
    image_files = sorted([f for f in os.listdir(folder_path) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    print(f"Found {len(image_files)} images")

    if not image_files:
        print("No valid images found in folder")
        sys.exit(1)

    # Process images
    total_error = 0
    count = 0
    
    for filename in image_files:
        try:
            image_path = os.path.join(folder_path, filename)
            print(f"\nProcessing: {image_path}")
            
            ground_truth = extract_ground_truth(filename)
            if ground_truth is None:
                print(f"Skipping {filename} - cannot extract ground truth")
                continue

            img_tensor = load_and_preprocess_image(image_path).to(device)
            print(f"Image tensor shape: {img_tensor.shape}")
            
            with torch.no_grad():
                output = model(img_tensor)
                predicted = output[0].cpu().numpy() * 60.0
                predicted = np.clip(predicted, -60, 60)

            error = np.mean(np.abs(predicted - ground_truth))
            total_error += error
            count += 1

            print(f"Predicted: [{predicted[0]:.1f}, {predicted[1]:.1f}]")
            print(f"Actual:    [{ground_truth[0]:.1f}, {ground_truth[1]:.1f}]")
            print(f"Error:     {error:.2f}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

    if count > 0:
        print(f"\nProcessed {count} images")
        print(f"Average Error: {total_error/count:.2f}")
    else:
        print("\nNo images were successfully processed")