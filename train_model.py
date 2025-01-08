import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
import numpy as np
from torch.utils.data import random_split
from PIL import Image

class FaceDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (string): Root directory path (data folder)
            split (string): 'train' or 'test'
            transform (callable, optional): Optional transform to be applied on images
        """
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.classes = os.listdir(self.root_dir)  # Get person names from folders
        
        # Create label encoder
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.classes)
        
        # Get all image paths and labels
        self.image_paths = []
        self.labels = []
        
        for person in self.classes:
            person_dir = os.path.join(self.root_dir, person)
            image_files = [f for f in os.listdir(person_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
            
            for img_file in image_files:
                self.image_paths.append(os.path.join(person_dir, img_file))
                self.labels.append(person)
                
        self.labels = self.label_encoder.transform(self.labels)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label

def train_model(data_dir, num_epochs=10, batch_size=4):
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Create datasets
    try:
        train_dataset = FaceDataset(data_dir, split='train', transform=transform)
        test_dataset = FaceDataset(data_dir, split='test', transform=transform)
    except Exception as e:
        raise Exception(f"Error loading datasets: {str(e)}")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(train_dataset.classes)
    model = FaceDetector(num_classes).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    best_accuracy = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Save model after each epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'label_encoder': train_dataset.label_encoder,
            'classes': train_dataset.classes
        }, 'face_recognition_model.pth')

    print('Finished Training')
    return model

class FaceDetector(nn.Module):
    def __init__(self, num_classes):
        super(FaceDetector, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    data_dir = "data"
    train_model(data_dir) 