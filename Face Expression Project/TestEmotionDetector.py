# import required packages 

import cv2 
import numpy as np
from keras.models import model_from_json
from tensorflow.keras.models import model_from_json
from tensorflow.keras import Sequential


# Emotion dictionary
emotion_dict = {
    0: "Angry",
    1: "Disgusted",
    2: "Fearful",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprised"
}


# Load JSON and create model

json_file_path = r"C:\Users\ayush\Desktop\Mini Project GEHU\model\emotion_model.json"
with open(json_file_path, 'r') as json_file:
    loaded_model_json = json_file.read()


# Specify custom objects

emotion_model = model_from_json(loaded_model_json, custom_objects={'Sequential': Sequential})


# Load weights into new model

weights_file_path = r"C:\Users\ayush\Desktop\Mini Project GEHU\model\emotion_model.weights.h5"
emotion_model.load_weights(weights_file_path)
print("Loaded model from disk")



# Load the Haar Cascade for face detection
face_cascade_path = r"C:\Users\ayush\Desktop\Mini Project GEHU\haarcascades\haarcascade_frontalface_default.xml"
face_detector = cv2.CascadeClassifier(face_cascade_path)

# Start the webcam feed
cap = cv2.VideoCapture(0)  


while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))
    
    if not ret:
        break
    
    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # Draw a scanning effect on the frame
    for i in range(0, frame.shape[1], 20):
        cv2.line(frame, (i, 0), (i, frame.shape[0]), (0, 255, 0), 1)

    for i in range(0, frame.shape[0], 20):
        cv2.line(frame, (0, i), (frame.shape[1], i), (0, 255, 0), 1)

    # Process each detected face
    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-58), (x+w, y+h+18), (0, 255, 0), 4)  # Draw a rectangle around the face

        # Extract the region of interest (ROI) for emotion detection
        roi_gray_frame = gray_frame[y:y+h, x:x+w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        # Make a prediction using your emotion
