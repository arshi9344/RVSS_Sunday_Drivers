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

torch.manual_seed(10)

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
k_folds = 10
num_epochs = 10
kf = KFold(n_splits=k_folds, shuffle=True, random_state=0)
loss_function = nn.CrossEntropyLoss()
net = Net()
net.apply(reset_weights)
optimizer = optim.Adam(net.parameters(), lr=1e-4)
results = {}

losses = {'train': [], 'val': []}
accs = {'train': [], 'val': []}
best_acc = 0
best_fold = -1

#######################################################################################################################################
####     TRAINING                                                                                                                  ####
#######################################################################################################################################

results = {}  # Initialize results dictionary

for fold, (train_ids, val_ids) in enumerate(kf.split(dataset)):
    print(f'FOLD {fold}')
    print('--------------------------------')
    
    train_subsampler = SubsetRandomSampler(train_ids)
    val_subsampler = SubsetRandomSampler(val_ids)
    
    trainloader = DataLoader(dataset, batch_size=8, sampler=train_subsampler)
    valloader = DataLoader(dataset, batch_size=1, sampler=val_subsampler)
    
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        epoch_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f'Epoch {epoch + 1} loss: {epoch_loss / len(trainloader)}')
        losses['train'].append(epoch_loss / len(trainloader))
        accs['train'].append(100. * correct / total)

        correct_pred = {classname: 0 for classname in class_labels}
        total_pred = {classname: 0 for classname in class_labels}

        val_loss = 0
        actual = []
        predicted = []
        with torch.no_grad():
            for data in valloader:
                images, labels = data
                outputs = net(images)
                _, predictions = torch.max(outputs, 1)
                loss = loss_function(outputs, labels)
                val_loss += loss.item()
                actual.extend(labels.tolist())
                predicted.extend(predictions.tolist())
                for label, prediction in zip(labels, predictions):
                    if label == prediction:
                        correct_pred[class_labels[label.item()]] += 1
                    total_pred[class_labels[label.item()]] += 1

        class_accs = []
        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            class_accs.append(accuracy)

        accs['val'].append(np.mean(class_accs))
        losses['val'].append(val_loss / len(valloader))

        if np.mean(class_accs) > best_acc:
            best_acc = np.mean(class_accs)
            best_fold = fold
            torch.save(net.state_dict(), 'best_model.pth')
    print('fold accuracy:', np.mean(class_accs))
    results[fold] = np.mean(class_accs)  # Store accuracy for this fold

#######################################################################################################################################
####     PERFORMANCE EVALUATION                                                                                                    ####
#######################################################################################################################################
print('Best Fold:', best_fold)
net.load_state_dict(torch.load(f'best_model.pth', weights_only=True))

correct = 0
total = 0
actual = []  # Reset actual list for each fold
predicted = []  # Reset predicted list for each fold
with torch.no_grad():
    for data in valloader:
            images, labels = data
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()
            actual.extend(labels.tolist())
            predicted.extend(predictions.tolist())

    print(f'Accuracy of the network on the {total} validation images: {100 * correct // total} %')

    cm = metrics.confusion_matrix(actual, predicted, normalize='true')
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot()
    plt.show()

    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f}%')

#######################################################################################################################################
####     PLOTTING METRICS                                                                                                          ####
#######################################################################################################################################

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(losses['train'], label='Training Loss')
plt.plot(losses['val'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss over Epochs')

plt.subplot(1, 2, 2)
plt.plot(accs['train'], label='Training Accuracy')
plt.plot(accs['val'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('Accuracy over Epochs')

plt.show()

print('--------------------------------')
print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
for key, value in results.items():
    print(f'Fold {key}: {value:.2f}%')
print(f'Average: {sum(results.values()) / len(results):.2f}%')

print(f'Best model was from fold {best_fold} with accuracy {best_acc:.2f}%')

