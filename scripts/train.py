import os
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Define image size
IMG_SIZE = 48  # FER2013 images are 48x48

# Emotion labels corresponding to the FER2013 dataset
emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']

# Function to load images from a directory
def load_images_from_folder(folder):
    images = []
    labels = []
    for label in emotion_labels:
        emotion_folder = os.path.join(folder, label)
        if os.path.exists(emotion_folder):
            for filename in os.listdir(emotion_folder):
                img_path = os.path.join(emotion_folder, filename)
                img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE), color_mode='grayscale')
                img_array = img_to_array(img)
                images.append(img_array)
                labels.append(emotion_labels.index(label))
    return np.array(images), np.array(labels)

# Load images from train and test folders inside FER2013
train_folder = r'D:\FACIAL EXPRESSION RECOGNITION\dataset\FER 2013\train'
test_folder = r'D:\FACIAL EXPRESSION RECOGNITION\dataset\FER 2013\test'

train_images, train_labels = load_images_from_folder(train_folder)
test_images, test_labels = load_images_from_folder(test_folder)

# Normalize the images
train_images = train_images / 255.0
test_images = test_images / 255.0

# One-hot encode the labels
train_labels = to_categorical(train_labels, num_classes=7)
test_labels = to_categorical(test_labels, num_classes=7)

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

# Create the model
model = models.Sequential()

# Add convolutional layers, max pooling layers, and fully connected layers
model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(7, activation='softmax'))  # 7 classes for the 7 emotions

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=30, batch_size=64, validation_data=(X_val, y_val))

# Save the model to a file
model.save('facial_expression_model.h5')

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"Test accuracy: {test_acc}")
