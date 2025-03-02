import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class FER2013Dataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.df.iloc[idx, 0])
        image = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (224, 224))  # Resize to EfficientNet input size (224x224)
        label = int(self.df.iloc[idx, 1])  # Get the emotion label

        if self.transform:
            image = self.transform(image)

        return image, label
import torch
import torch.nn as nn
from torch.optim import Adam
from efficientnet_pytorch import EfficientNet

# Define the model
class FERModel(nn.Module):
    def __init__(self, num_classes=7):
        super(FERModel, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b0')
        self.model._fc = nn.Linear(self.model._fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Instantiate model, loss function, and optimizer
model = FERModel(num_classes=7)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

# Define DataLoader (you can change the directory and other parameters as needed)
train_dataset = FER2013Dataset(csv_file='train.csv', img_dir='train', transform=transforms.ToTensor())
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Training loop (simplified)
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.cuda(), labels.cuda()

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}, Accuracy: {correct/total}")

# Save the model
torch.save(model.state_dict(), 'efficientnet_fer_model.pth')
