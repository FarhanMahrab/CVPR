import cv2 as cv
print(cv.__version__)
import numpy as np
from collections import deque
from tensorflow.keras.models import load_model

print("Loading model...")
model = load_model('F:/Final/Assignment 2/face_recognition_model.keras')
print("Model loaded successfully!")

# Load class labels
class_names = ['Farhan', 'Hamim', 'Saad', 'Shishir']
print(f"Classes: {class_names}")

# Load face cascade
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

if face_cascade.empty():
    print("Error: Could not load face cascade classifier")
    exit()
else:
    print("Face cascade loaded successfully!")

# Define padding percentage
padding_ratio = 0.2

# Initialize webcam
webcam = cv.VideoCapture(0)

if not webcam.isOpened():
    print("Error: Could not open webcam")
    exit()
else:
    print("Webcam opened successfully!")

# Set webcam properties for better performance
webcam.set(cv.CAP_PROP_FRAME_WIDTH, 640)
webcam.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

# Queue for smoothing predictions
prediction_queue = deque(maxlen=5)

print("\nStarting face recognition...")
print("Press 'q' to quit")

while True:
    # Capture frame-by-frame
    ret, frame = webcam.read()
    
    if not ret:
        print("Error: Failed to capture frame")
        break
    
    # Convert the frame to grayscale for face detection
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.equalizeHist(gray)  # Normalize lighting

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.2, 
        minNeighbors=7, 
        minSize=(50, 50)
    )

    # Loop over detected faces
    for (x, y, w, h) in faces:
        # Calculate padding
        pad_x = int(w * padding_ratio)
        pad_y = int(h * padding_ratio)

        # Expand the bounding box
        x1 = max(x - pad_x, 0)
        y1 = max(y - pad_y, 0)
        x2 = min(x + w + pad_x, frame.shape[1])
        y2 = min(y + h + pad_y, frame.shape[0])

        # Extract the face
        face_img = frame[y1:y2, x1:x2]
        
        # Skip if face image is empty
        if face_img.size == 0:
            continue

        # Preprocess the face image
        face_img_resized = cv.resize(face_img, (256, 256))
        face_img_array = np.expand_dims(face_img_resized / 255.0, axis=0)

        # Get predictions
        predictions = model.predict(face_img_array, verbose=0)
        predicted_class = np.argmax(predictions)
        confidence = np.max(predictions)

        # Add to the queue for smoothing
        prediction_queue.append(predicted_class)
        
        # Get most common prediction from queue
        if prediction_queue:
            most_common_prediction = max(set(prediction_queue), key=prediction_queue.count)
        else:
            most_common_prediction = predicted_class

        # Determine class name and color based on confidence
        if confidence > 0.8:
            class_name = class_names[most_common_prediction]
            color = (0, 255, 0)  # Green
        else:
            class_name = 'Unknown'
            color = (0, 0, 255)  # Red

        # Draw the bounding box and label
        cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv.putText(frame, f'{class_name} ({confidence:.2f})', 
                   (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 
                   0.9, color, 2)

    # Show the video frame
    cv.imshow('Webcam Face Recognition', frame)

    # Break if 'q' is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        print("\nQuitting...")
        break

# Release resources
webcam.release()
cv.destroyAllWindows()
print("Program ended.")