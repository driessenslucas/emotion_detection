import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set the path to your dataset
home_path = './'
data_dir = os.path.join(home_path, 'emotion detection/')

# Define transformations for training and validation sets
train_transforms = transforms.Compose([
    transforms.Resize((128, 128)),        # Resize images
    transforms.RandomHorizontalFlip(),   # Randomly flip images
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Random brightness/contrast changes
    transforms.RandomAffine(degrees=0, shear=0.2),  # Shear transformation
    transforms.ToTensor(),               # Convert to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
])

val_transforms = transforms.Compose([
    transforms.Resize((128, 128)),        # Resize images
    transforms.ToTensor(),               # Convert to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
])

def load_data(data_dir):
    """Load and split the dataset into training and validation sets."""
    # Load the full dataset
    full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transforms)

    # Split the dataset into training and validation sets
    train_size = int(0.7 * len(full_dataset))  # 70% for training
    val_size = len(full_dataset) - train_size  # 30% for validation
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Apply validation transforms to the validation dataset
    val_dataset.dataset.transform = val_transforms

    # Create DataLoader for training and validation sets
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    return train_loader, val_loader

class EmotionDetectionModel(nn.Module):
    """Define the Emotion Detection Model."""
    def __init__(self, num_classes=3):
        super(EmotionDetectionModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)

        self.global_pool = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        self.fc1 = nn.Linear(128, 256)  # Input size changed from 256 to 128
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        """Forward pass through the network."""
        # Initial Conv Block
        x = self.pool(self.dropout(F.relu(self.bn1(self.conv1(x)))))
        x = self.pool(self.dropout(F.relu(self.bn2(self.conv2(x)))))

        # Second Conv Block
        x = self.pool(self.dropout(F.relu(self.bn3(self.conv3(x)))))
        x = self.pool(self.dropout(F.relu(self.bn4(self.conv4(x)))))

        # Third Conv Block
        x = self.pool(self.dropout(F.relu(self.bn5(self.conv5(x)))))
        x = self.pool(self.dropout(F.relu(self.bn6(self.conv6(x)))))

        # Global Average Pooling and Dense Layers
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)

        return x

def train_model(model, train_loader, val_loader, num_epochs=50):
    """Train the model."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    best_val_accuracy = 0.0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)  # Move data to the device
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_predictions, val_labels = [], []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)  # Move data to the device
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_predictions.extend(predicted.cpu().numpy())  # Move predictions to CPU for processing
                val_labels.extend(labels.cpu().numpy())  # Move labels to CPU for processing

        # Calculate validation accuracy
        val_accuracy = accuracy_score(val_labels, val_predictions)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}, Validation Loss: {val_loss/len(val_loader):.4f}, Validation Accuracy: {val_accuracy:.4f}')

        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_emotion_model.pth')

    print("Training Complete.")

def evaluate_model(model, val_loader, class_names):
    """Evaluate the model performance."""
    model.eval()  # Set the model to evaluation mode
    val_predictions, val_labels = [], []
    
    with torch.no_grad():  # Disable gradient tracking
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)  # Move data to the device
            outputs = model(images)  # Forward pass
            _, predicted = torch.max(outputs, 1)  # Get the predicted classes
            val_predictions.extend(predicted.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
    
    # Calculate accuracy
    val_accuracy = accuracy_score(val_labels, val_predictions)
    print(f'Validation Accuracy: {val_accuracy:.4f}')

    # Print classification report for more metrics
    print("Classification Report:")
    print(classification_report(val_labels, val_predictions, target_names=class_names))

    # Optional: Plot confusion matrix
    cm = confusion_matrix(val_labels, val_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_loader, val_loader = load_data(data_dir)

    # Initialize model and move to device
    model = EmotionDetectionModel()
    model.to(device)
    print(f"Using device: {device}")

    # Train the model
    train_model(model, train_loader, val_loader)

    # Define class names for evaluation
    class_names = ['angry', 'happy', 'sad']  # Adjust this based on your dataset

    # Evaluate the model
    evaluate_model(model, val_loader, class_names)
