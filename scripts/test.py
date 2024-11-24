import cv2
# import pyttsx3
from tensorflow.keras.models import load_model
import numpy as np

# Load the pre-trained model
model = load_model('facial_expression_model.h5')

# Initialize the text-to-speech engine
# engine = pyttsx3.init()

# Emotion labels corresponding to the FER2013 dataset
emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']

# Start webcam
cap = cv2.VideoCapture(0)

# Start processing the webcam feed
while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Convert the frame to grayscale (as the model expects grayscale images)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using OpenCV's built-in Haar Cascade Classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48), flags=cv2.CASCADE_SCALE_IMAGE)

    # Process each detected face
    for (x, y, w, h) in faces:
        # Crop and resize the face to the required 48x48
        face_region = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face_region, (48, 48))
        face_resized = face_resized.reshape(1, 48, 48, 1)
        face_resized = face_resized / 255.0  # Normalize

        # Predict the facial expression
        prediction = model.predict(face_resized)
        predicted_class = np.argmax(prediction, axis=1)

        # Map the predicted class to the corresponding emotion label
        predicted_emotion = emotion_labels[predicted_class[0]]

        # Draw a rectangle around the face and put the predicted emotion label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Speak out the detected emotion
        # engine.say(f"Detected: {predicted_emotion}")
        # engine.runAndWait()

    # Display the frame with the detected face and emotion label
    cv2.imshow("Facial Expression Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
