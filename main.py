import cv2
import os
import tensorflow as tf
import numpy as np
from pygame import mixer
import serial
import time
import threading

# Initialize alarm sound
mixer.init()
sound = mixer.Sound('alarm.wav')

# Load Haar Cascade models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# Load the trained model
model = tf.keras.models.load_model(os.path.join("model", "model.keras"))

# Labels
lbl = ['Close', 'Open']

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Font and score
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
score = 0

# Initialize serial communication
arduino = serial.Serial(port='COM11', baudrate=115200, timeout=0.1)

def write_read(value):
    """Send data to Arduino."""
    arduino.write(bytes(value, 'utf-8'))
    time.sleep(0.05)

# Process interval (to skip frames)
frame_count = 0
process_interval = 2  # Process every 2nd frame

# Threading for prediction
prediction_lock = threading.Lock()
prediction_result = None

def predict_eye_state(eye_frame):
    """Run the prediction in a separate thread."""
    global prediction_result
    eye = eye_frame / 255.0  # Normalize
    eye = eye.reshape(1, 80, 80, 3)
    with prediction_lock:
        prediction_result = model.predict(eye)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Skip frames to reduce computational load
    if frame_count % process_interval == 0:
        frame_height, frame_width = frame.shape[:2]
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray_frame, scaleFactor=1.2, minNeighbors=5, minSize=(50, 50)
        )

        # Draw rectangles around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Detect eyes
        eyes = eye_cascade.detectMultiScale(
            gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)
        )

        for (ex, ey, ew, eh) in eyes:
            eye_frame = frame[ey:ey + eh, ex:ex + ew]
            eye_frame = cv2.resize(eye_frame, (80, 80))

            # Run prediction in a separate thread
            threading.Thread(target=predict_eye_state, args=(eye_frame,)).start()

    # Handle prediction result
    with prediction_lock:
        if prediction_result is not None:
            if prediction_result[0][0] > 0.30:  # Close condition
                score += 1
                cv2.putText(frame, "Closed", (10, frame_height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
                if score > 5:
                    write_read('1')  # Trigger Arduino alarm
            elif prediction_result[0][1] > 0.70:  # Open condition
                score = max(score - 1, 0)
                cv2.putText(frame, "Open", (10, frame_height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.putText(frame, f'Score: {score}', (100, frame_height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Drowsiness Detection', frame)

    # Quit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

# Release resources
cap.release()
cv2.destroyAllWindows()
