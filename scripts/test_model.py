import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn as nn

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

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Test model in PyTorch')
    parser.add_argument('--model_path', type=str, default='best_model.pth', help='Path to the model weights')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run the model (cpu or cuda)')
    args = parser.parse_args()

    device = torch.device(args.device)
    model = load_model(args.model_path, device)
    img_tensor = load_and_preprocess_image(args.image_path)
    img_tensor = img_tensor.to(device)

    print(f"Input image tensor shape: {img_tensor.shape}")
    with torch.no_grad():
        output_tensor = model(img_tensor)
        speeds = output_tensor[0].cpu().numpy() * 60.0  # Denormalize speeds
    print(f"Calculated Speeds: {speeds}")