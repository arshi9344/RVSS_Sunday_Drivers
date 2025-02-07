import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SubsetRandomSampler, ConcatDataset, WeightedRandomSampler, Dataset
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
from model import Net


def seed_everything(random_seed: int):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    random.seed(random_seed)

torch.backends.cudnn.benchmark = True
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
datasets_path = os.path.join('.', 'data', 'Datasets_4_train')

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

def imshow(img):
    img = img / 2 + 0.5 #unnormalize
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    plt.imshow(npimg)
    plt.show()

# Update visualization code
example_ims, example_lbls = next(iter(trainloader_plot))
imshow(torchvision.utils.make_grid(example_ims))
plt.axis('off')
plt.savefig('training_validation.png')
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
plt.hist(left_speeds, bins=40, color='blue', alpha=0.7, range=(min(left_speeds), max(left_speeds)))
plt.xlabel('Left Wheel Speed')
plt.ylabel('Count')
plt.title('Left Wheel Speed Distribution')

# Right speeds distribution  
plt.subplot(1, 2, 2)
plt.hist(right_speeds, bins=40, color='red', alpha=0.7, range=(min(right_speeds), max(right_speeds)))
plt.xlabel('Right Wheel Speed')
plt.ylabel('Count')
plt.title('Right Wheel Speed Distribution')

plt.tight_layout()
plt.savefig('speed_distributions.png')
plt.show()

print(f"Speed ranges:")
print(f"Left: {min(left_speeds):.2f} to {max(left_speeds):.1f}")
print(f"Right: {min(right_speeds):.2f} to {max(right_speeds):.1f}")

# Calculate histogram with appropriate range for normalized speeds
left_counts, left_bins = np.histogram(left_speeds, bins=20, range=(0, 1))
right_counts, right_bins = np.histogram(right_speeds, bins=20, range=(0, 1))

# Add smoothing factor to avoid zero weights
epsilon = 1e-7
left_counts = left_counts + epsilon
right_counts = right_counts + epsilon

# Calculate weights
# left_weights = 1.0 / np.sqrt(left_counts)
# right_weights = 1.0 / np.sqrt(right_counts)
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
# sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

# Before the k-fold loop, load all data to GPU
all_images = []
all_speeds = []
for i in range(len(dataset)):
    img, speed = dataset[i]
    all_images.append(img)
    all_speeds.append(speed)

all_images = torch.stack(all_images).to(device)
all_speeds = torch.stack(all_speeds).to(device)

class InMemoryDataset(Dataset):
    def __init__(self, images, speeds):
        self.images = images
        self.speeds = speeds
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.speeds[idx]

# Create in-memory dataset
gpu_dataset = InMemoryDataset(all_images, all_speeds)

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
seeds = [10, 42, 123, 256, 789]
best_loss = [float('inf')] * k_folds
initial_lr=1e-3

# Initialize results dictionary with both MSE and MAE
results = {f'fold_{i}': {'mse': 0, 'mae': 0} for i in range(k_folds)}


for fold, (train_ids, val_ids) in enumerate(kf.split(dataset)):
    print(f'FOLD {fold}')
    print('--------------------------------')
    
    # Data setup
    train_weights = samples_weight[train_ids]
    train_sampler = WeightedRandomSampler(train_weights, len(train_ids), replacement=True)

    # Validation: use exact indices without weighting
    val_sampler = SubsetRandomSampler(val_ids)
    
    # Modify your data loaders
    trainloader = DataLoader(
        gpu_dataset, 
        batch_size=512,
        sampler=train_sampler,
        pin_memory=False,  # Not needed as data is already on GPU
        num_workers=0,     # Not needed as data is in memory
        persistent_workers=False
    )

    valloader = DataLoader(
        gpu_dataset, 
        batch_size=1024,
        sampler=val_sampler,
        pin_memory=False,
        num_workers=0,
        persistent_workers=False
    )
    
    # Model setup
    seed_everything(seeds[fold])
    net = Net().to(device)
    net.apply(reset_weights)
    
    #optimizer = torch.optim.Adam(net.parameters(), lr=initial_lr)
    optimizer = torch.optim.RMSprop(net.parameters(), lr=initial_lr)
    #optimizer = optim.SGD(net.parameters(), lr=initial_lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, min_lr=1e-6)
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
                images, speeds = data[0].to(device), data[1].float().to(device)
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
        if val_mse < best_loss[fold]:  # Use square brackets for list indexing
            best_loss[fold] = val_mse
            best_fold = fold
            torch.save(net.state_dict(), f'model_{fold}.pth')

        scheduler.step(val_loss)
        print(f'Current LR: {optimizer.param_groups[0]["lr"]}')

        print(f'Epoch {epoch + 1} - Training MSE: {train_mse:.4f}, '
            f'Validation MSE: {val_mse:.4f}, '
            f'Training MAE: {train_mae:.4f}, '
            f'Validation MAE: {val_mae:.4f}')
        
    # After the 'for epoch in range(num_epochs):' loop finishes, plot this fold's stats.
    train_mse_list = losses[f'train_fold_{fold}']
    val_mse_list = losses[f'val_fold_{fold}']
    train_mae_list = accs[f'train_fold_{fold}']
    val_mae_list = accs[f'val_fold_{fold}']

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Plot MSE
    epochs = range(1, len(train_mse_list) + 1)
    axes[0].plot(epochs, train_mse_list, label='Train MSE', linestyle='--')
    axes[0].plot(epochs, val_mse_list, label='Val MSE')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE Loss')
    axes[0].set_title(f'Fold {fold} - MSE')
    axes[0].legend()
    axes[0].grid(True)
    # Set upper limit to 95th percentile of data
    axes[0].set_ylim(bottom=0, top=np.percentile( train_mse_list + val_mse_list, 95))

    # Plot MAE
    axes[1].plot(epochs, train_mae_list, label='Train MAE', linestyle='--')
    axes[1].plot(epochs, val_mae_list, label='Val MAE')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].set_title(f'Fold {fold} - MAE')
    axes[1].legend()
    axes[1].grid(True)
    # Set upper limit to 95th percentile of data
    axes[1].set_ylim(bottom=0, top=np.percentile(train_mae_list + val_mae_list, 95))

    plt.tight_layout()
    plt.savefig(f"Fold_{fold}_stats.png")
    plt.show(block=False)

    # Store fold results
    results[fold] = {
        'mse': val_mse,
        'mae': val_mae
    }

    

#######################################################################################################################################
####     PERFORMANCE EVALUATION                                                                                                    ####
#######################################################################################################################################
print('Best Fold:', best_fold)
net.load_state_dict(torch.load(f'model_{fold}.pth', weights_only=True))

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
# print(f'Best model from fold {best_fold} with MSE {best_loss:.4f}')
