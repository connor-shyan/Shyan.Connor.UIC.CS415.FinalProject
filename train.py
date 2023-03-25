#
# CS 415 Final Project - Exercise Pose Recognition (Solution 2)
# train.py - For extracting training data from dataset
# by Connor Shyan (UIN 650119775)
# UIC, Fall 2022
#


#
# Importing necessary libraries
#
import mediapipe as mp
import cv2
import pandas as pd
import os


#
# Function for extracting training data from dataset
#
def build_dataset(path, dataset_type):

    # Getting column names for csv file
    data = []
    for p in points:
        x = str(p)[13:]
        data.append(x + "_x")       # x-coordinates of pose landmarks
        data.append(x + "_y")       # y-coordinates of pose landmarks
        data.append(x + "_z")       # z-coordinates of pose landmarks
        data.append(x + "_vis")     # visibility of pose landmarks
    data.append("target")           # exercise pose name
    data = pd.DataFrame(columns=data)  # formatting the data columns
    count = 0

    # Going through the training dataset folder for images
    dirnames = [x[1] for x in os.walk(path)][0]
    for k in range(len(dirnames)):
        for img in os.listdir(path + "/" + dirnames[k]):
            temp = []

            # Processing the images and extracting pose landmarks data
            img = cv2.imread(path + "/" + dirnames[k] + "/" + img)
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(imgRGB)
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                for i, j in zip(points, landmarks):
                    temp = temp + [j.x, j.y, j.z, j.visibility]

                # Labeling the exercise pose name to data and continuing
                temp.append(dirnames[k])
                data.loc[count] = temp
                count += 1

    # Saving the extracted pose landmarks data as a csv file
    data.to_csv(dataset_type + ".csv")


# Necessary variables for extracting pose landmarks data
mpPose = mp.solutions.pose
pose = mpPose.Pose()
points = mpPose.PoseLandmark

# Building the csv file for the training dataset from images folder
build_dataset("CS415FinalDataset", "cs415final")
