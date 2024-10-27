# import cv2
# import numpy as np
# from tensorflow import keras

# # Load the trained emotion detection model
# model = keras.models.load_model('./best_emotion_model.pth')  # Ensure the model path is correct

# # Print all available webcams (optional)
# for i in range(10):
#     cap = cv2.VideoCapture(i)
#     if cap.isOpened():
#         print(f"Webcam found at index: {i}")
#         cap.release()

# # Initialize the webcam or video source
# cap = cv2.VideoCapture(0)  # Change the index if the default camera is not at index 0

# # Check if the camera is opened successfully
# if not cap.isOpened():
#     print("Error: Could not open camera.")
#     exit()

# # Load the Haar Cascade classifier for face detection
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# # Define a function to preprocess the detected face for the model
# def preprocess_face(face):
#     # Preprocess the face as needed, e.g., resize and normalize
#     face = cv2.resize(face, (64, 64))  # Resize to match the model input size
#     face = face / 255.0  # Normalize to [0, 1]
#     face = np.expand_dims(face, axis=-1)  # Add channel dimension if needed (depends on model input shape)
#     return face

# while True:
#     ret, frame = cap.read()

#     # Check if the frame is empty
#     if not ret:
#         print("Error: Could not read frame.")
#         break
    
#     # Perform face detection and emotion prediction for each frame
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    
#     for (x, y, w, h) in faces:
#         # Extract the detected face
#         detected_face = frame[y:y + h, x:x + w]
        
#         # Preprocess the detected face and pass it to the emotion detection model
#         face = preprocess_face(detected_face)
#         face = np.expand_dims(face, axis=0)  # Add a batch dimension
#         emotion_probs = model.predict(face)

#         # Get the predicted emotion class
#         predicted_emotion = ['angry', 'happy', 'sad'][np.argmax(emotion_probs)]
        
#         # Draw a rectangle around the detected face
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
#         # Draw the emotion prediction on the frame
#         cv2.putText(frame, f"Emotion: {predicted_emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
#     # Display the frame with emotion prediction
#     cv2.imshow('Emotion Detector', frame)
    
#     if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' key to exit
#         break

# # Release the video capture and close all OpenCV windows
# cap.release()
# cv2.destroy


import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18
from sklearn.metrics import accuracy_score
import torch.nn.functional as F

# Define your EmotionDetectionModel here if it's not imported
class EmotionDetectionModel(nn.Module):
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
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
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

# Load the trained emotion detection model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionDetectionModel(num_classes=3)  # Make sure this matches your training setup
model.load_state_dict(torch.load('./best_emotion_model.pth'))  # Load the model
model.to(device)  # Move the model to the appropriate device
model.eval()  # Set the model to evaluation mode

# Initialize the webcam or video source
cap = cv2.VideoCapture(0)  # Change the index if necessary

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define a function to preprocess the detected face for the model
def preprocess_face(face):
    face = cv2.resize(face, (128, 128))  # Resize to match model input size
    face = face / 255.0  # Normalize to [0, 1]
    face = np.transpose(face, (2, 0, 1))  # Change to (C, H, W)
    face_tensor = torch.FloatTensor(face)  # Convert to PyTorch tensor
    face_tensor = face_tensor.unsqueeze(0)  # Add batch dimension
    return face_tensor.to(device)  # Move to the appropriate device

while True:
    ret, frame = cap.read()

    # Check if the frame is empty
    if not ret:
        print("Error: Could not read frame.")
        break
    
    # Perform face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        # Extract the detected face
        detected_face = frame[y:y + h, x:x + w]
        
        # Preprocess the detected face and pass it to the emotion detection model
        face = preprocess_face(detected_face)
        
        with torch.no_grad():  # Disable gradient calculation for inference
            outputs = model(face)  # Forward pass
            _, predicted = torch.max(outputs, 1)  # Get predicted class

        # Get the predicted emotion class
        predicted_emotion = ['angry', 'happy', 'happy'][predicted.item()]
        
        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Draw the emotion prediction on the frame
        cv2.putText(frame, f"Emotion: {predicted_emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display the frame with emotion prediction
    cv2.imshow('Emotion Detector', frame)
    
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' key to exit
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
