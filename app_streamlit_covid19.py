from torchvision import datasets,transforms

path = r"E:/4th semester/python/code/Covid19-dataset/train"

transformation = transforms.Compose( [
    transforms.Resize((128,128)),
    transforms.ToTensor() ]
)

dataset = datasets.ImageFolder(path,transform=transformation)
dataset

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Dataset path
path = r"E:/4th semester/python/code/Covid19-dataset/train"

# Transformations
transformation = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Load dataset
dataset = datasets.ImageFolder(path, transform=transformation)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define CNN
class SimpleCNN(nn.Module):
    def __init__(self, in_channels):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=5, stride=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=3)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32*7*7, 1080)
        self.fc2 = nn.Linear(1080, 512)
        self.fc3 = nn.Linear(512, 32)
        self.fc4 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)  # no sigmoid here
        return x

# Get input channels
img, label = dataset[0]
in_channels = img.shape[0]

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model, loss, optimizer
model = SimpleCNN(in_channels).to(device)
criterion = nn.BCEWithLogitsLoss()  # handles sigmoid internally
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10

# Training loop
for epoch in range(1, num_epochs + 1):
    total_loss = 0
    model.train()
    for x, y in dataloader:
        x, y = x.to(device), y.view(-1, 1).float().to(device)  # reshape y to match output
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch}/{num_epochs} - Loss: {total_loss:.4f}")

#muticlass classification CNN

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Load MNIST dataset
dataset = datasets.MNIST(root='E:/4th semester/python/code/Covid19-dataset/train',
                         train=True,
                         download=True,
                         transform=transforms.ToTensor())

dataloader = DataLoader(dataset, shuffle=True, batch_size=64)
img, label = dataset[0]

in_channel, width, height = img.shape
output = len(dataset.classes)

# Define Simple CNN
class SimpleCNN(nn.Module):
    def __init__(self, in_channel, width, height, output):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 16, kernel_size=5, stride=1)
        self.pool1 = nn.AvgPool2d(kernel_size=3)  # Avg Pooling
        
        self.flatten = nn.Flatten()
        
        # Calculate dimensions after convolution and pooling
        conv_output_w = (width + 2*1 - 5) // 1 + 1
        conv_output_h = (height + 2*1 - 5) // 1 + 1
        pool_output_w = (conv_output_w - 3) // 3 + 1
        pool_output_h = (conv_output_h - 3) // 3 + 1

        self.fc1 = nn.Linear(16 * pool_output_w * pool_output_h, 64)
        self.fc2 = nn.Linear(64, output)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.conv1(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize model, loss function and optimizer
model = SimpleCNN(in_channel, width, height, output)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
num_epoch = 10

# Training loop
for epoch in range(0, num_epoch+1):
    total_loss = 0
    model.train()
    
    for x, y in dataloader:
        output = model(x)
        loss = criterion(output, y)
        total_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch [{epoch}/{num_epoch}] Loss: {total_loss}")


#muticlass classification CNN

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Load MNIST dataset
dataset = datasets.MNIST(root='E:/4th semester/python/code/Covid19-dataset/train',
                         train=True,
                         download=True,
                         transform=transforms.ToTensor())

dataloader = DataLoader(dataset, shuffle=True, batch_size=64)
img, label = dataset[0]

in_channel, width, height = img.shape
output = len(dataset.classes)

# Define Simple CNN
class SimpleCNN(nn.Module):
    def __init__(self, in_channel, width, height, output):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 16, kernel_size=5, stride=1)
        self.pool1 = nn.AvgPool2d(kernel_size=3)  # Avg Pooling
        
        self.flatten = nn.Flatten()
        
        # Calculate dimensions after convolution and pooling
        conv_output_w = (width + 2*1 - 5) // 1 + 1
        conv_output_h = (height + 2*1 - 5) // 1 + 1
        pool_output_w = (conv_output_w - 3) // 3 + 1
        pool_output_h = (conv_output_h - 3) // 3 + 1

        self.fc1 = nn.Linear(16 * pool_output_w * pool_output_h, 64)
        self.fc2 = nn.Linear(64, output)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.conv1(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize model, loss function and optimizer
model = SimpleCNN(in_channel, width, height, output)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
num_epoch = 100

# Training loop
for epoch in range(0, num_epoch+1):
    total_loss = 0
    model.train()
    
    for x, y in dataloader:
        output = model(x)
        loss = criterion(output, y)
        total_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch [{epoch}/{num_epoch}] Loss: {total_loss}")


#muticlass classification CNN --- test dataset 

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Load MNIST dataset
dataset = datasets.MNIST(root='E:/4th semester/python/code/Covid19-dataset/test',
                         train=True,
                         download=True,
                         transform=transforms.ToTensor())

dataloader = DataLoader(dataset, shuffle=True, batch_size=64)
img, label = dataset[0]

in_channel, width, height = img.shape
output = len(dataset.classes)

# Define Simple CNN
class SimpleCNN(nn.Module):
    def __init__(self, in_channel, width, height, output):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 16, kernel_size=5, stride=1)
        self.pool1 = nn.AvgPool2d(kernel_size=3)  # Avg Pooling
        
        self.flatten = nn.Flatten()
        
        # Calculate dimensions after convolution and pooling
        conv_output_w = (width + 2*1 - 5) // 1 + 1
        conv_output_h = (height + 2*1 - 5) // 1 + 1
        pool_output_w = (conv_output_w - 3) // 3 + 1
        pool_output_h = (conv_output_h - 3) // 3 + 1

        self.fc1 = nn.Linear(16 * pool_output_w * pool_output_h, 64)
        self.fc2 = nn.Linear(64, output)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.conv1(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize model, loss function and optimizer
model = SimpleCNN(in_channel, width, height, output)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
num_epoch = 100

# Training loop
for epoch in range(0, num_epoch+1):
    total_loss = 0
    model.train()
    
    for x, y in dataloader:
        output = model(x)
        loss = criterion(output, y)
        total_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch [{epoch}/{num_epoch}] Loss: {total_loss}")

# resnet18 model --- train dataset

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os

path = r"E:/4th semester/python/code/Covid19-dataset/train"

transformation = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

class FilteredImageFolder(datasets.ImageFolder):
    def find_classes(self, directory):
        classes, class_to_idx = super().find_classes(directory)
        # remove MNIST if present
        if "MNIST" in classes:
            classes.remove("MNIST")
        class_to_idx = {cls: i for i, cls in enumerate(classes)}
        return classes, class_to_idx
# Load dataset
data = FilteredImageFolder(path, transform=transformation)
loader = DataLoader(data, batch_size=32, shuffle=True)
class_names = data.classes
print("✅ Classes used:", class_names)
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
# Freeze convolutional layers
for param in model.parameters():
    param.requires_grad = False
# Replace final layer (multi-class classification)
num_classes = len(class_names)
model.fc = nn.Linear(model.fc.in_features, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)
num_epochs = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
for epoch in range(num_epochs):
    total_loss = 0
    correct = 0
    total = 0
    model.train()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        # Compute loss
        loss = criterion(outputs, y)
        # Backprop
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        # Accuracy
        preds = outputs.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    acc = 100 * correct / total
    print(f"Epoch {epoch+1}/{num_epochs} - "
          f"Loss: {total_loss:.4f} - Accuracy: {acc:.2f}%")
#resnet50 model -- train dataset

import sklearn
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------
# Dataset path & transforms
# -------------------
path = r"E:/4th semester/python/code/Covid19-dataset/train"

transformation = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

# -------------------
# Custom ImageFolder to filter out MNIST class
# -------------------
class FilteredImageFolder(datasets.ImageFolder):
    def find_classes(self, directory):
        classes, class_to_idx = super().find_classes(directory)
        if "MNIST" in classes:
            classes.remove("MNIST")
        class_to_idx = {cls: i for i, cls in enumerate(classes)}
        return classes, class_to_idx

# Load dataset
data = FilteredImageFolder(path, transform=transformation)
loader = DataLoader(data, batch_size=32, shuffle=True)
class_names = data.classes
print("✅ Classes used:", class_names)
num_classes = len(class_names)

# -------------------
# Model: ResNet-50
# -------------------
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
# Freeze convolutional layers
for param in model.parameters():
    param.requires_grad = False
# Replace final layer
model.fc = nn.Linear(model.fc.in_features, num_classes)

# -------------------
# Loss, optimizer, device
# -------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    
    acc = 100 * correct / total
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss:.4f} - Accuracy: {acc:.2f}%")
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        outputs = model(x)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# Metrics
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

print(f"\n✅ Evaluation Metrics:")
print(f"Accuracy: {accuracy*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall: {recall*100:.2f}%")
print(f"F1-score: {f1*100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


#resnet50 model -- test dataset

import sklearn
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------
# Dataset path & transforms
# -------------------
path = r"E:/4th semester/python/code/Covid19-dataset/test"

transformation = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

# -------------------
# Custom ImageFolder to filter out MNIST class
# -------------------
class FilteredImageFolder(datasets.ImageFolder):
    def find_classes(self, directory):
        classes, class_to_idx = super().find_classes(directory)
        if "MNIST" in classes:
            classes.remove("MNIST")
        class_to_idx = {cls: i for i, cls in enumerate(classes)}
        return classes, class_to_idx

# Load dataset
data = FilteredImageFolder(path, transform=transformation)
loader = DataLoader(data, batch_size=32, shuffle=True)
class_names = data.classes
print("✅ Classes used:", class_names)
num_classes = len(class_names)

# -------------------
# Model: ResNet-50
# -------------------
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
# Freeze convolutional layers
for param in model.parameters():
    param.requires_grad = False
# Replace final layer
model.fc = nn.Linear(model.fc.in_features, num_classes)

# -------------------
# Loss, optimizer, device
# -------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    
    acc = 100 * correct / total
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss:.4f} - Accuracy: {acc:.2f}%")
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        outputs = model(x)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# Metrics
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

print(f"\n✅ Evaluation Metrics:")
print(f"Accuracy: {accuracy*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall: {recall*100:.2f}%")
print(f"F1-score: {f1*100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


#VGG -16 model - train dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------
# Dataset path & transforms
# -------------------
path = r"E:/4th semester/python/code/Covid19-dataset/train"

transformation = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

# -------------------
# Custom ImageFolder to filter out MNIST class
# -------------------
class FilteredImageFolder(datasets.ImageFolder):
    def find_classes(self, directory):
        classes, class_to_idx = super().find_classes(directory)
        if "MNIST" in classes:
            classes.remove("MNIST")
        class_to_idx = {cls: i for i, cls in enumerate(classes)}
        return classes, class_to_idx

# Load dataset
data = FilteredImageFolder(path, transform=transformation)
loader = DataLoader(data, batch_size=32, shuffle=True)
class_names = data.classes
print("✅ Classes used:", class_names)
num_classes = len(class_names)

# -------------------
# Model: VGG16
# -------------------
model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
# Freeze convolutional layers
for param in model.features.parameters():
    param.requires_grad = False
# Replace classifier's final layer
model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

# -------------------
# Loss, optimizer, device
# -------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier[6].parameters(), lr=1e-3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# -------------------
# Training loop
# -------------------
num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    
    acc = 100 * correct / total
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss:.4f} - Accuracy: {acc:.2f}%")

# -------------------
# Evaluation
# -------------------
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        outputs = model(x)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# Metrics
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

print(f"\n✅ Evaluation Metrics:")
print(f"Accuracy: {accuracy*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall: {recall*100:.2f}%")
print(f"F1-score: {f1*100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


#VGG -16 model - test dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------
# Dataset path & transforms
# -------------------
path = r"E:/4th semester/python/code/Covid19-dataset/test"

transformation = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

# -------------------
# Custom ImageFolder to filter out MNIST class
# -------------------
class FilteredImageFolder(datasets.ImageFolder):
    def find_classes(self, directory):
        classes, class_to_idx = super().find_classes(directory)
        if "MNIST" in classes:
            classes.remove("MNIST")
        class_to_idx = {cls: i for i, cls in enumerate(classes)}
        return classes, class_to_idx

# Load dataset
data = FilteredImageFolder(path, transform=transformation)
loader = DataLoader(data, batch_size=32, shuffle=True)
class_names = data.classes
print("✅ Classes used:", class_names)
num_classes = len(class_names)

# -------------------
# Model: VGG16
# -------------------
model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
# Freeze convolutional layers
for param in model.features.parameters():
    param.requires_grad = False
# Replace classifier's final layer
model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

# -------------------
# Loss, optimizer, device
# -------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier[6].parameters(), lr=1e-3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# -------------------
# Training loop
# -------------------
num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    
    acc = 100 * correct / total
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss:.4f} - Accuracy: {acc:.2f}%")

# -------------------
# Evaluation
# -------------------
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        outputs = model(x)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# Metrics
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

print(f"\n✅ Evaluation Metrics:")
print(f"Accuracy: {accuracy*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall: {recall*100:.2f}%")
print(f"F1-score: {f1*100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

torch.save(model.state_dict(), os.path.join(os.getcwd(), "vgg16_model.pth"))


import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image

# -----------------------------
# Load your trained model
# -----------------------------
model = torch.load("vgg16_full_model.pth", map_location="cpu")  # if you saved full model
model.eval()

# Define transforms (same as training)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Class labels
classes = ["COVID-19", "Viral Pneumonia", "Normal"]

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🩺 COVID-19 Chest X-ray Classifier")
st.write("Upload a chest X-ray image to predict COVID-19, Viral Pneumonia, or Normal.")

uploaded_file = st.file_uploader("Choose an X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    img_tensor = transform(image).unsqueeze(0)

    # Make prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)

    st.success(f"Prediction: **{classes[predicted.item()]}**")
