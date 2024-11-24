# Real-Time Facial Expression Detection using Deep Learning

This repository contains a solution for real-time facial expression detection using a Convolutional Neural Network (CNN) trained on the **FER2013** dataset. The model classifies facial expressions into seven categories: **Anger**, **Disgust**, **Fear**, **Happiness**, **Sadness**, **Surprise**, and **Neutral**. The system uses OpenCV for real-time webcam processing and integrates text-to-speech feedback for detected emotions.

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Install Dependencies](#install-dependencies)
- [Prepare Data](#prepare-data)
- [Training the Model](#training-the-model)
- [Running the Training Script](#running-the-training-script)
- [Testing the Model](#testing-the-model)
- [Running the Testing Script](#running-the-testing-script)
- [Installation](#installation)

## Project Overview

This project aims to leverage deep learning to classify facial expressions from video input in real-time. The **FER2013** dataset, widely used for emotion recognition tasks, contains grayscale images of faces labeled with different emotions. The approach consists of training a CNN to learn the features of facial expressions and applying this trained model to detect emotions from the webcam feed.

### Key Features:
- **Real-time emotion recognition** from a webcam feed.
- **Text-to-speech feedback** for detected facial expressions.
- **Custom-trained CNN** using the FER2013 dataset.
- **OpenCV-based face detection** for locating faces in the webcam feed.

## Install Dependencies
Install the required Python libraries by running the following command:

                    pip install -r requirements.txt

## Prepare Data
Ensure the FER2013 dataset is organized correctly into the train and test directories. These folders should contain subfolders for each emotion label, such as Anger, Disgust, Fear, etc.

You can download the dataset from Kaggle's FER2013 dataset page.

## Training the Model
To train the facial expression detection model, run the train.py script. This script:

- Loads the FER2013 training images from the train folder.
- Preprocesses the images by resizing and normalizing.
- Trains a CNN to classify the facial expressions.
- Saves the trained model as facial_expression_model.h5.

## Running the Training Script
Execute the following command to start training:

                      python train.py

The training process will:

- Split the training data into training and validation sets.
- Build and train the CNN architecture with layers for feature extraction and classification.
- Output training accuracy and validation accuracy after each epoch.
- Save the trained model to the file facial_expression_model.h5.

## Testing the Model
Once the model is trained, you can test it in real-time using the webcam. The test.py script will:

- Capture video frames from the webcam.
- Detect faces in each frame using OpenCV's Haar Cascade Classifier.
- Classify the facial expressions using the trained CNN.
- Display the detected emotion on the video feed and provide text-to-speech feedback.

## Running the Testing Script
Run the following command to start real-time emotion detection:

                    python test.py

To stop the webcam feed, press the 'q' key.

## Installation
To set up and run the project locally, follow the steps below:

            git clone https://github.com/psroy007//real-time-facial-expression-detection.git
