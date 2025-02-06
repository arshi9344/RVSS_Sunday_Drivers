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

import matplotlib.pyplot as plt

from steerDS import SteerDataSet

torch.manual_seed(10)

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
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((60, 60)),
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



# Verify data loading
# sample_image, sample_speeds = dataset[0]
# print(f"Sample image shape: {sample_image.shape}")
# print(f"Sample speeds: Left={sample_speeds[0]}, Right={sample_speeds[1]}")
# print(f"Dataset size: {len(dataset)} images")

trainloader_plot = DataLoader(
    dataset, 
    batch_size=32,
    shuffle=True,
    num_workers=2
)

# Extract all labels
all_y = [label for _, label in dataset]

# Add denormalization function
def denormalize(tensor):
    mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    return tensor * std + mean

# Update visualization code
example_ims, example_lbls = next(iter(trainloader_plot))
plt.figure(figsize=(15, 5))
grid = torchvision.utils.make_grid(denormalize(example_ims))
plt.imshow(grid.permute(1, 2, 0).clamp(0, 1))
plt.axis('off')
plt.savefig('training_metrics.png')
plt.show()

# Extract all speeds
left_speeds = []
right_speeds = []
for i in range(len(dataset)):
    _, speeds = dataset[i]
    left_speeds.append(speeds[0].item())
    right_speeds.append(speeds[1].item())

# # Create figure with two subplots
# plt.figure(figsize=(12, 5))

# # Left speeds distribution
# plt.subplot(1, 2, 1)
# plt.hist(left_speeds, bins=20, color='blue', alpha=0.7)
# plt.xlabel('Left Wheel Speed')
# plt.ylabel('Count')
# plt.title('Left Wheel Speed Distribution')

# # Right speeds distribution  
# plt.subplot(1, 2, 2)
# plt.hist(right_speeds, bins=20, color='red', alpha=0.7)
# plt.xlabel('Right Wheel Speed')
# plt.ylabel('Count')
# plt.title('Right Wheel Speed Distribution')

# plt.tight_layout()
# plt.savefig('speed_distributions.png')
# plt.show()

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
        self.conv1 = nn.Conv2d(3, 16, 3)
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
num_epochs = 10
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

# Initialize results dictionary with both MSE and MAE
results = {f'fold_{i}': {'mse': 0, 'mae': 0} for i in range(k_folds)}

for fold, (train_ids, val_ids) in enumerate(kf.split(dataset)):
    print(f'FOLD {fold}')
    print('--------------------------------')
    
    # Data setup
    train_subsampler = SubsetRandomSampler(train_ids)
    val_subsampler = SubsetRandomSampler(val_ids)
    trainloader = DataLoader(dataset, batch_size=64, sampler=sampler)
    valloader = DataLoader(dataset, batch_size=1, sampler=sampler)
    
    # Model setup
    net = Net().to(device)
    net.apply(reset_weights)
    initial_lr = 1e-4
    optimizer = torch.optim.Adam(net.parameters(), lr=initial_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
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
            inputs, speeds = data[0].to(device), data[1].to(device)
            speeds = speeds.float()
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_function(outputs, speeds)
            loss.backward()
            optimizer.step()
            
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
                speeds = speeds.float()
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
            torch.save(net.state_dict(), 'best_model.pth')

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
net.load_state_dict(torch.load(f'best_model.pth', weights_only=True))

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

# After all folds complete, create plots
plt.figure(figsize=(12, 8))

# Plot MSE for all folds
plt.subplot(2, 1, 1)
for fold in range(k_folds):
    plt.plot(losses[f'train_fold_{fold}'], '--', label=f'Train MSE Fold {fold}')
    plt.plot(losses[f'val_fold_{fold}'], label=f'Val MSE Fold {fold}')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.grid(True)
plt.title('MSE Loss Across All Folds')

# Plot MAE for all folds
plt.subplot(2, 1, 2)
for fold in range(k_folds):
    plt.plot(accs[f'train_fold_{fold}'], '--', label=f'Train MAE Fold {fold}')
    plt.plot(accs[f'val_fold_{fold}'], label=f'Val MAE Fold {fold}')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.grid(True)
plt.title('MAE Across All Folds')

plt.tight_layout()
plt.show()
plt.savefig('training_stats.png')
plt.close()

if is_running_in_ssh():
    plt.savefig('training_stats.png')
    print("Plots saved as training_stats.png")
    plt.close()
else:
    plt.show()

print('--------------------------------')
print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
for fold, metrics in results.items():
    print(f'Fold {fold}: MSE={metrics["mse"]:.4f}, MAE={metrics["mae"]:.4f}')
print(f'Average MSE: {sum(m["mse"] for m in results.values()) / len(results):.4f}')
print(f'Average MAE: {sum(m["mae"] for m in results.values()) / len(results):.4f}')
print(f'Best model from fold {best_fold} with MSE {best_loss:.4f}')