import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SubsetRandomSampler, ConcatDataset
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

torch.manual_seed(42)

def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

#######################################################################################################################################
####     SETTING UP THE DATASET                                                                                                    ####
#######################################################################################################################################

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((20, 60)),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ])

script_path = os.path.dirname(os.path.realpath(__file__))
dataset_train_part = SteerDataSet(os.path.join(script_path, '..', 'data', 'train_starter'), '.jpg', transform)
dataset_val_part = SteerDataSet(os.path.join(script_path, '..', 'data', 'val_starter'), '.jpg', transform)
dataset = ConcatDataset([dataset_train_part, dataset_val_part])
print("The dataset contains %d images " % len(dataset))

# Combine class labels from both datasets
class_labels = list(set(dataset_train_part.class_labels + dataset_val_part.class_labels))

#######################################################################################################################################
####     INITIALISE OUR NETWORK                                                                                                    ####
#######################################################################################################################################

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(384, 256)
        self.fc2 = nn.Linear(256, 5)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

#######################################################################################################################################
####     K-FOLD CROSS VALIDATION                                                                                                  ####
#######################################################################################################################################
## setup the k-fold cross validation
k_folds = 5
num_epochs = 30
kf = KFold(n_splits=k_folds, shuffle=True, random_state=0)
loss_function = nn.CrossEntropyLoss()

results = {}

# Initialize metrics dictionaries with fold-specific keys
losses = {f'train_fold_{i}': [] for i in range(k_folds)}
losses.update({f'val_fold_{i}': [] for i in range(k_folds)})
accs = {f'train_fold_{i}': [] for i in range(k_folds)}
accs.update({f'val_fold_{i}': [] for i in range(k_folds)})

best_acc = 0
best_fold = -1

#######################################################################################################################################
####     TRAINING                                                                                                                  ####
#######################################################################################################################################

results = {}  # Initialize results dictionary

for fold, (train_ids, val_ids) in enumerate(kf.split(dataset)):
    print(f'FOLD {fold}')
    print('--------------------------------')
    
    # Data setup
    train_subsampler = SubsetRandomSampler(train_ids)
    val_subsampler = SubsetRandomSampler(val_ids)
    trainloader = DataLoader(dataset, batch_size=8, sampler=train_subsampler)
    valloader = DataLoader(dataset, batch_size=1, sampler=val_subsampler)
    
    # Model setup
    net = Net()
    net.apply(reset_weights)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

    # Training loop
    for epoch in range(num_epochs):
        net.train()  # Set to training mode
        epoch_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(trainloader, 0):
            # Training step
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Training metrics
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Validation phase
        net.eval()  # Set to evaluation mode
        val_correct = 0
        val_total = 0
        val_loss = 0
        with torch.no_grad():
            for data in valloader:
                images, labels = data
                outputs = net(images)
                _, predictions = torch.max(outputs, 1)
                loss = loss_function(outputs, labels)
                val_loss += loss.item()
                val_total += labels.size(0)
                val_correct += (predictions == labels).sum().item()

        # Record metrics
        fold_accuracy = 100.0 * val_correct / val_total
        if fold_accuracy > best_acc:
            best_acc = fold_accuracy
            best_fold = fold
            torch.save(net.state_dict(), 'best_model.pth')
        
        # In the training loop, update the metrics storage
        losses[f'train_fold_{fold}'].append(epoch_loss/len(trainloader))
        accs[f'train_fold_{fold}'].append(100.0 * correct / total)
        losses[f'val_fold_{fold}'].append(val_loss/len(valloader))
        accs[f'val_fold_{fold}'].append(fold_accuracy)

        print(f'Epoch {epoch + 1} - Training loss: {epoch_loss/len(trainloader):.4f}, '
              f'Validation accuracy: {fold_accuracy:.2f}%')

    # Store fold results
    results[fold] = fold_accuracy

#######################################################################################################################################
####     PERFORMANCE EVALUATION                                                                                                    ####
#######################################################################################################################################
print('Best Fold:', best_fold)
net.load_state_dict(torch.load(f'best_model.pth', weights_only=True))

correct = 0
total = 0
actual = []  # Reset actual list for each fold
predicted = []  # Reset predicted list for each fold
correct_pred = {classname: 0 for classname in class_labels}
total_pred = {classname: 0 for classname in class_labels}

with torch.no_grad():
    for data in valloader:
            images, labels = data
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()
            actual.extend(labels.tolist())
            predicted.extend(predictions.tolist())
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[class_labels[label]] += 1
                total_pred[class_labels[label]] += 1

    print(f'Accuracy of the network on the {total} validation images: {100 * correct // total} %')

    cm = metrics.confusion_matrix(actual, predicted, normalize='true')
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot()
    plt.show()

    # for classname, correct_count in correct_pred.items():
    #     accuracy = 100 * float(correct_count) / total_pred[classname]
    #     print(f'Accuracy for class: {classname:5s} is {accuracy:.1f}%')

#######################################################################################################################################
####     PLOTTING METRICS                                                                                                          ####
#######################################################################################################################################

# Create a figure with k_folds rows and 2 columns
plt.figure(figsize=(15, 5*k_folds))

for fold in range(k_folds):
    # Training Loss subplot
    plt.subplot(k_folds, 2, 2*fold + 1)
    plt.plot(losses[f'train_fold_{fold}'], label='Training Loss')
    plt.plot(losses[f'val_fold_{fold}'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'Loss over Epochs - Fold {fold}')

    # Accuracy subplot
    plt.subplot(k_folds, 2, 2*fold + 2)
    plt.plot(accs[f'train_fold_{fold}'], label='Training Accuracy')
    plt.plot(accs[f'val_fold_{fold}'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title(f'Accuracy over Epochs - Fold {fold}')

plt.tight_layout()
plt.show()

print('--------------------------------')
print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
for key, value in results.items():
    print(f'Fold {key}: {value:.2f}%')
print(f'Average: {sum(results.values()) / len(results):.2f}%')

print(f'Best model was from fold {best_fold} with accuracy {best_acc:.2f}%')

