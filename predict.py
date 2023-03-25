#
# CS 415 Final Project - Exercise Pose Recognition (Solution 2)
# predict.py - For pose prediction
# by Connor Shyan (UIN 650119775)
# UIC, Fall 2022
#


#
# Importing necessary libraries
#
import mediapipe as mp
import cv2
import pandas as pd
from sklearn.svm import SVC


#
# Function to predict exercise pose from video input
#
def predict_video(model, video=0, show=False):

    # Get video input (webcam input by default)
    cap = cv2.VideoCapture(video)
    while cap.isOpened():
        temp = []

        # Reading image from the video frames
        success, img = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Processing image to extract pose landmarks data
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(img)
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            for j in landmarks:
                temp = temp + [j.x, j.y, j.z, j.visibility]

            # Predict the pose using trained model and display it
            y = model.predict([temp])
            name = str(y[0])
            if show:
                mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
                (w, h), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
                cv2.rectangle(img, (40, 40), (40 + w, 60), (255, 255, 255), cv2.FILLED)
                cv2.putText(img, name, (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
                cv2.imshow("Exercise Pose Prediction", img)
                if cv2.waitKey(5) & 0xFF == 27:
                    break
    cap.release()


# Training the prediction model from pose landmarks data in csv file
data_train = pd.read_csv("cs415final.csv")
data_train = data_train.drop(labels="Unnamed: 0", axis=1)
X, Y = data_train.iloc[:, :data_train.shape[1] - 1], data_train['target']
model = SVC(kernel='poly', decision_function_shape='ovo')
model.fit(X, Y)

# Necessary variables for pose prediction and drawing keypoints
mpPose = mp.solutions.pose
pose = mpPose.Pose(static_image_mode=True, min_detection_confidence=0.2)
mpDraw = mp.solutions.drawing_utils

# Predicting the exercise pose from video input
predict_video(model, show=True)
