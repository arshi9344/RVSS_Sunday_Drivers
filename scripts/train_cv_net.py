import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SubsetRandomSampler, ConcatDataset, WeightedRandomSampler
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np
import sklearn.metrics as metrics
from sklearn.model_selection import KFold
import random

import matplotlib.pyplot as plt

from steerDS import SteerDataSet

def seed_everything(random_seed: int):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    random.seed(random_seed)

# torch.backends.cudnn.benchmark = True
# torch.cuda.set_per_process_memory_fraction(0.95)  # Use 95% of available memory

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f'Using GPU: {torch.cuda.get_device_name(0)}')
else:
    print('Using CPU')

def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

def is_running_in_ssh():
    return 'SSH_CLIENT' in os.environ or 'SSH_TTY' in os.environ

#######################################################################################################################################
####     SETTING UP THE DATASET                                                                                                    ####
#######################################################################################################################################
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((120, 120)),
    transforms.ColorJitter(
        brightness=0.2,  # Brightness variation of ±20%
        contrast=0.2,    # Contrast variation of ±20%
        saturation=0.2,  # Saturation variation of ±20%
        hue=0.1         # Slight hue shifts (±10% of total hue range)
    ),
    # Random adjustments to lighting
    transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.15),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

script_path = os.path.dirname(os.path.realpath(__file__))
datasets_path = os.path.join(script_path, '..', 'data', 'datasets_4_training')

# Debug directory structure
print(f"Looking for datasets in: {datasets_path}")
if not os.path.exists(datasets_path):
    raise FileNotFoundError(f"Dataset path not found: {datasets_path}")

# Get all subdirectories
dataset_folders = [f.path for f in os.scandir(datasets_path) if f.is_dir()]
print(f"Found {len(dataset_folders)} subdirectories:")
for folder in dataset_folders:
    print(f"- {folder}")

if not dataset_folders:
    raise ValueError(f"No dataset folders found in: {datasets_path}")

# Load each dataset with validation
datasets = []
total_images = 0
for folder in dataset_folders:
    try:
        dataset = SteerDataSet(folder, '.jpg', transform)
        num_images = len(dataset)
        if num_images > 0:
            datasets.append(dataset)
            total_images += num_images
            print(f"Loaded {num_images} images from {os.path.basename(folder)}")
        else:
            print(f"Warning: No jpg images found in {folder}")
    except Exception as e:
        print(f"Error loading dataset from {folder}: {e}")

if total_images == 0:
    raise ValueError("No valid images found in any dataset folder")

# Combine datasets
dataset = ConcatDataset(datasets)
print(f"Total dataset contains {len(dataset)} images")

# Verify dataset before creating loader
print(f"Dataset size: {len(dataset)}")
sample_idx = 0
try:
    sample = dataset[sample_idx]
    #print(f"Sample {sample_idx} shape: {sample[0].shape}")
except Exception as e:
    print(f"Failed to load sample {sample_idx}: {str(e)}")

trainloader_plot = DataLoader(
    dataset, 
    batch_size=32,
    shuffle=True,
    num_workers=2,
    drop_last=True  # Avoid partial batches
)

# Extract all labels
all_y = [label for _, label in dataset]

# Update visualization code
example_ims, example_lbls = next(iter(trainloader_plot))
imshow(torchvision.utils.make_grid(example_ims))
plt.axis('off')
plt.savefig('training_validation.png')
plt.show()
plt.close()

# Extract all speeds
left_speeds = []
right_speeds = []
for i in range(len(dataset)):
    _, speeds = dataset[i]
    left_speeds.append(speeds[0].item())
    right_speeds.append(speeds[1].item())

# Create figure with two subplots
plt.figure(figsize=(12, 5))

# Left speeds distribution
plt.subplot(1, 2, 1)
plt.hist(left_speeds, bins=20, color='blue', alpha=0.7)
plt.xlabel('Left Wheel Speed')
plt.ylabel('Count')
plt.title('Left Wheel Speed Distribution')

# Right speeds distribution  
plt.subplot(1, 2, 2)
plt.hist(right_speeds, bins=20, color='red', alpha=0.7)
plt.xlabel('Right Wheel Speed')
plt.ylabel('Count')
plt.title('Right Wheel Speed Distribution')

plt.tight_layout()
plt.savefig('speed_distributions.png')
plt.show()

print(f"Speed ranges:")
print(f"Left: {min(left_speeds):.1f} to {max(left_speeds):.1f}")
print(f"Right: {min(right_speeds):.1f} to {max(right_speeds):.1f}")

# Calculate histogram with smoothing
left_counts, left_bins = np.histogram(left_speeds, bins=20, range=(min(left_speeds), max(left_speeds)))
right_counts, right_bins = np.histogram(right_speeds, bins=20, range=(min(right_speeds), max(right_speeds)))

# Add smoothing factor to avoid zero weights
epsilon = 1e-7
left_counts = left_counts + epsilon
right_counts = right_counts + epsilon

# Calculate weights
left_weights = 1.0 / left_counts
right_weights = 1.0 / right_counts

# Normalize weights
left_weights = left_weights / np.sum(left_weights)
right_weights = right_weights / np.sum(right_weights)

# Assign weights with bounds checking
samples_weight = []
for i in range(len(dataset)):
    _, speeds = dataset[i]
    left_idx = min(max(np.digitize(speeds[0].item(), left_bins) - 1, 0), len(left_weights) - 1)
    right_idx = min(max(np.digitize(speeds[1].item(), right_bins) - 1, 0), len(right_weights) - 1)
    weight = max((left_weights[left_idx] + right_weights[right_idx]) / 2, epsilon)
    samples_weight.append(weight)

samples_weight = torch.FloatTensor(samples_weight)

# Create weighted sampler
sampler = WeightedRandomSampler(samples_weight, len(samples_weight))


#######################################################################################################################################
####     INITIALISE OUR NETWORK                                                                                                    ####
#######################################################################################################################################

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 1) Convolution + Pooling block
        self.conv1 = nn.Conv2d(3, 16, 3)  # RGB version
        # self.conv1 = nn.Conv2d(1, 16, 3)    # Grayscale version
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        
        # 2) Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2)

        # 3) Fully connected
        #    Flattened dimension after conv/pool is 1600, so fc1 in_features=1600
        self.fc1 = nn.Linear(1600, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 2)  # final 2 outputs (e.g. left/right speeds)

        # Optional: You can define activation once and reuse, or inline them in forward
        self.relu = nn.ReLU()

    def forward(self, x):
        # input shape: [batch_size, 3, 60, 60]

        # Conv1 -> ReLU -> Pool => [batch_size, 16, 29, 29]
        x = self.pool(self.relu(self.conv1(x)))
        
        # Conv2 -> ReLU -> Pool => [batch_size, 32, 13, 13]
        x = self.pool(self.relu(self.conv2(x)))
        
        # Conv3 -> ReLU -> Pool => [batch_size, 64, 5, 5]
        x = self.pool(self.relu(self.conv3(x)))

        # Flatten => [batch_size, 64*5*5 = 1600]
        x = torch.flatten(x, start_dim=1)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        out = self.fc3(x)  # shape: [batch_size, 2]

        return out
    #     # Convolutional layers (Feature Extraction)
    #     self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
    #     self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
    #     self.conv3 = nn.Conv2d(16, 32, kernel_size=3)
    #     self.conv4 = nn.Conv2d(32, 64, kernel_size=3)
    #     self.conv5 = nn.Conv2d(64, 64, kernel_size=3)

    #     # Pooling layer (Downsampling)
    #     self.pool = nn.MaxPool2d(kernel_size=2) #, stride=2)  # Reduce spatial dimensions by 2x

    #     # Fully Connected Layers (Decision Making)
    #     self.fc1 = nn.Linear(64, 256)  # Adjusted to match output feature map size
    #     self.fc2 = nn.Linear(256, 128)  # Added extra FC layer for better decision-making
    #     self.fc3 = nn.Linear(128, 2)  # Change to 2 outputs for left/right speeds
        
    #     # Activation function
    #     self.relu = nn.ReLU()

    # def forward(self, x):
    #     x = self.pool(self.relu(self.conv1(x)))  # Conv1 → ReLU → MaxPool
    #     x = self.pool(self.relu(self.conv2(x)))  # Conv2 → ReLU → MaxPool
    #     x = self.relu(self.conv3(x))  # Conv3 → ReLU
    #     x = self.relu(self.conv4(x))  # Conv4 → ReLU
    #     x = self.relu(self.conv5(x))  # Conv5 → ReLU

    #     # Flatten before feeding into fully connected layers
    #     x = torch.flatten(x, start_dim=1)  # Flatten all dimensions except batch

    #     # Fully Connected Layers + Activation
    #     x = self.relu(self.fc1(x))  # FC1 → ReLU
    #     x = self.relu(self.fc2(x))  # FC2 → ReLU
    #     output = self.fc3(x)  # Now outputs (left_speed, right_speed)

    #     return output

    # def __init__(self):
    #     super().__init__()
    #     self.conv1 = nn.Conv2d(3, 6, 5)
    #     self.conv2 = nn.Conv2d(6, 16, 5)
    #     self.pool = nn.MaxPool2d(2, 2)
    #     self.fc1 = nn.Linear(384, 256)
    #     self.fc2 = nn.Linear(256, 5)
    #     self.relu = nn.ReLU()
    
    # def forward(self, x):
    #     x = self.pool(self.relu(self.conv1(x)))
    #     x = self.pool(self.relu(self.conv2(x)))
    #     x = torch.flatten(x, 1)
    #     x = self.fc1(x)
    #     x = self.relu(x)
    #     x = self.fc2(x)
    #     return x


#######################################################################################################################################
####     K-FOLD CROSS VALIDATION                                                                                                  ####
#######################################################################################################################################
## setup the k-fold cross validation
k_folds = 5
num_epochs = 100
kf = KFold(n_splits=k_folds, shuffle=True, random_state=0)
#loss_function = nn.CrossEntropyLoss()
loss_function = nn.MSELoss()  # For regression task

results = {}

# Initialize metrics dictionaries with fold-specific keys
losses = {f'train_fold_{i}': [] for i in range(k_folds)}
losses.update({f'val_fold_{i}': [] for i in range(k_folds)})
accs = {f'train_fold_{i}': [] for i in range(k_folds)}
accs.update({f'val_fold_{i}': [] for i in range(k_folds)})

best_acc = 0
best_fold = -1
best_loss = float('inf')

#######################################################################################################################################
####     TRAINING                                                                                                                  ####
#######################################################################################################################################

# SEEDED TRAINING. Use different seeds! cals seed
seed = 10
seed_everything(seed)

# Initialize results dictionary with both MSE and MAE
results = {f'fold_{i}': {'mse': 0, 'mae': 0} for i in range(k_folds)}

for fold, (train_ids, val_ids) in enumerate(kf.split(dataset)):
    print(f'FOLD {fold}')
    print('--------------------------------')
    
    # Data setup
    train_subsampler = SubsetRandomSampler(train_ids)
    val_subsampler = SubsetRandomSampler(val_ids)
    trainloader = DataLoader(
        dataset, 
        batch_size=256, 
        sampler=sampler,
        pin_memory=True,  # Speeds up data transfer to GPU
        num_workers=8,   # Increased worker count
        persistent_workers=True  # Keep workers alive between epochs
    )
    valloader = DataLoader(
        dataset, 
        batch_size=512,    # Increased from 1 for better GPU utilization
        sampler=sampler,
        pin_memory=True,
        num_workers=8,
        persistent_workers=True
    )
    
    # Model setup
    net = Net().to(device)
    net.apply(reset_weights)
    initial_lr = 1e-4
    optimizer = torch.optim.Adam(net.parameters(), lr=initial_lr)
    #optimizer = torch.optim.RMSprop(net.parameters(), lr=initial_lr)
    #optimizer = optim.SGD(net.parameters(), lr=initial_lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    scaler = torch.amp.GradScaler('cuda')
    # print(f'TEST2') # Debugging
    # Training loop
    for epoch in range(num_epochs):
        net.train()  # Set to training mode
        epoch_loss = 0.0
        correct = 0
        total = 0
        mae = 0.0
        
        # Inside training loop
        epoch_mae = 0.0
        epoch_loss = 0.0
        total = 0

        for i, data in enumerate(trainloader, 0):
            inputs, speeds = data
            inputs = inputs.to(device, non_blocking=True)    # non_blocking=True for async transfer
            speeds = speeds.float().to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = net(inputs)
                loss = loss_function(outputs, speeds)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Accumulate batch metrics
            epoch_loss += loss.item()
            batch_mae = F.l1_loss(outputs, speeds, reduction='sum').item()
            epoch_mae += batch_mae
            total += speeds.size(0)

        # Calculate average training metrics
        train_mse = epoch_loss / len(trainloader)
        train_mae = epoch_mae / total

        # Record training metrics
        losses[f'train_fold_{fold}'].append(train_mse)
        accs[f'train_fold_{fold}'].append(train_mae)

        # Validation phase
        net.eval()  # Set to evaluation mode
        val_loss = 0.0
        val_mae = 0.0
        val_total = 0

        with torch.no_grad():
            for data in valloader:
                images, speeds = data[0].to(device), data[1].to(device)
                speeds = speeds.float().to(device)
                outputs = net(images)
                val_loss += loss_function(outputs, speeds).item()
                val_mae += F.l1_loss(outputs, speeds, reduction='sum').item()
                val_total += speeds.size(0)

        # Calculate average losses
        val_mse = val_loss / len(valloader)
        val_mae = val_mae / val_total

        # Record metrics
        losses[f'val_fold_{fold}'].append(val_mse)
        accs[f'val_fold_{fold}'].append(val_mae)

        # Calculate MSE
        train_mse = epoch_loss/len(trainloader)
        val_mse = val_loss/len(valloader)

        # Calculate MAE
        train_mae = F.l1_loss(outputs, speeds)
        val_mae = F.l1_loss(outputs, speeds)

        # Record metrics
        losses[f'train_fold_{fold}'].append(train_mse)
        losses[f'val_fold_{fold}'].append(val_mse)
        accs[f'train_fold_{fold}'].append(train_mae.item())  # Using MAE instead of accuracy
        accs[f'val_fold_{fold}'].append(val_mae.item())

        # Save best model based on validation MSE
        if val_mse < best_loss:  # Change from accuracy to loss
            best_loss = val_mse
            best_fold = fold
            torch.save(net.state_dict(), f'best_model_{seed}.pth')

        scheduler.step(val_loss)
        print(f'Current LR: {optimizer.param_groups[0]["lr"]}')

        print(f'Epoch {epoch + 1} - Training MSE: {train_mse:.4f}, '
              f'Validation MSE: {val_mse:.4f}, '
              f'Training MAE: {train_mae:.4f}, '
              f'Validation MAE: {val_mae:.4f}')

    # Store fold results
    results[fold] = {
        'mse': val_mse,
        'mae': val_mae
    }

#######################################################################################################################################
####     PERFORMANCE EVALUATION                                                                                                    ####
#######################################################################################################################################
print('Best Fold:', best_fold)
net.load_state_dict(torch.load(f'best_model_{seed}.pth', weights_only=True))

correct = 0
total = 0
actual = []  # Reset actual list for each fold
predicted = []  # Reset predicted list for each fold
correct_pred = {speed: 0 for speed in speeds}
total_pred = {speed: 0 for speed in speeds}

with torch.no_grad():
    total_error = 0
    for data in valloader:
        images, speeds = data[0].to(device), data[1].to(device)
        outputs = net(images)
        error = torch.abs(outputs - speeds).mean()
        total_error += error.item()
    
    avg_error = total_error / len(valloader)
    print(f'Average error on validation set: {avg_error:.4f}')

#######################################################################################################################################
####     PLOTTING METRICS                                                                                                          ####
#######################################################################################################################################

# Debug print to verify data
print("Plotting metrics...")
print(f"Number of folds: {k_folds}")
for fold in range(k_folds):
    print(f"Fold {fold} data lengths - Train MSE: {len(losses[f'train_fold_{fold}'])}, Val MSE: {len(losses[f'val_fold_{fold}'])}")

# Clear any existing plots
plt.clf()
plt.close('all')

# Create new figure with larger size and higher DPI
fig = plt.figure(figsize=(12, 8), dpi=100)

try:
    # Create a figure with k_folds rows and 2 columns
    fig, axes = plt.subplots(k_folds, 2, figsize=(12, 4*k_folds))
    
    for fold in range(k_folds):
        # Plot MSE
        train_loss = losses[f'train_fold_{fold}']
        val_loss = losses[f'val_fold_{fold}']
        epochs = range(1, len(train_loss) + 1)
        
        axes[fold, 0].plot(epochs, train_loss, '--', label='Train MSE')
        axes[fold, 0].plot(epochs, val_loss, label='Val MSE')
        axes[fold, 0].set_xlabel('Epoch')
        axes[fold, 0].set_ylabel('Mean Squared Error')
        axes[fold, 0].legend()
        axes[fold, 0].grid(True)
        axes[fold, 0].set_title(f'MSE Loss - Fold {fold}')

        # Plot MAE
        train_acc = accs[f'train_fold_{fold}']
        val_acc = accs[f'val_fold_{fold}']
        epochs = range(1, len(train_acc) + 1)
        
        axes[fold, 1].plot(epochs, train_acc, '--', label='Train MAE')
        axes[fold, 1].plot(epochs, val_acc, label='Val MAE')
        axes[fold, 1].set_xlabel('Epoch')
        axes[fold, 1].set_ylabel('Mean Absolute Error')
        axes[fold, 1].legend()
        axes[fold, 1].grid(True)
        axes[fold, 1].set_title(f'MAE - Fold {fold}')

    plt.tight_layout()
    plt.savefig('training_stats.png', bbox_inches='tight', pad_inches=0.1)
    print("Plot saved as training_stats.png")

except Exception as e:
    print(f"Error during plotting: {str(e)}")

finally:
    plt.close('all')

print('--------------------------------')
print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
for fold, metrics in results.items():
    print(f'Fold {fold}: MSE={metrics["mse"]:.4f}, MAE={metrics["mae"]:.4f}')
print(f'Average MSE: {sum(m["mse"] for m in results.values()) / len(results):.4f}')
print(f'Average MAE: {sum(m["mae"] for m in results.values()) / len(results):.4f}')
print(f'Best model from fold {best_fold} with MSE {best_loss:.4f}')
