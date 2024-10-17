import cv2
import numpy as np
from tensorflow import keras

# Load the trained emotion detection model
model = keras.models.load_model('./emotiondetect_models/emotion_detection_v2.h5')  # Replace with your model file


#print all available webcams
for i in range(10):
    cap = cv2.VideoCapture(i)


# Initialize the webcam or video source
cap = cv2.VideoCapture(0)  # Try changing this index if the default camera is not at index 0


# Check if the camera is opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Load the Haar Cascade classifier for face detection (you can replace this with MTCNN or other methods)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Define a function to preprocess the detected face for the new model
def preprocess_face(face):
    # Preprocess the face as needed, e.g., resize and normalize
    face = cv2.resize(face, (64, 64))  # Resize to match the model input size
    face = face / 255.0  # Normalize to [0, 1]
    return face

while True:
    ret, frame = cap.read()

    # Check if the frame is empty
    if not ret:
        print("Error: Could not read frame.")
        break
    
    # Perform face detection and emotion prediction for each frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        # Extract the detected face
        detected_face = frame[y:y + h, x:x + w]
        
        # Preprocess the detected face and pass it to the emotion detection model
        face = preprocess_face(detected_face)
        face = np.expand_dims(face, axis=0)  # Add a batch dimension
        emotion_probs = model.predict(face)
        
        # Get the predicted emotion class
        predicted_emotion = ['angry', 'happy', 'sad'][np.argmax(emotion_probs)]
        
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