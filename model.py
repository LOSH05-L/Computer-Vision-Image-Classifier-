import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 1. Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using compute device: {device}")

# 2. Data Transforms & Augmentations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Assuming a dataset directory structure: dataset/train/class_a, dataset/train/class_b
train_dir = "./dataset/train"
if os.path.exists(train_dir):
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    num_classes = len(train_dataset.classes)
    print(f"Detected classes: {train_dataset.classes}")
else:
    print("Warning: './dataset/train' directory not found. Please populate dataset for training.")

# 3. Custom CNN Architecture
class ImageClassifierCNN(nn.Module):
    def __init__(self, num_classes):
        super(ImageClassifierCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 64x64
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 32x32
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 16x16
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# 4. Training Loop Function
def train_model(epochs=10):
    if not os.path.exists(train_dir):
        return
    
    model = ImageClassifierCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Starting training loop...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc * 100:.2f}%")

    # Save trained model weights
    torch.save(model.state_dict(), "cnn_model_weights.pth")
    print("Model training complete and weights saved to cnn_model_weights.pth")

if __name__ == "__main__":
    train_model(epochs=5)