This code was written for the final project of CS 415 (Computer Vision) at the University of Illinois at Chicago (UIC). This is the second solution for our team's topic of "Exercise Pose Recognition" mainly worked on by Connor Shyan, while Lucas Jang mainly worked on the first solution.

As with first solution, the second solution relies on OpenCV and MediaPipe Pose to extract pose landmarks data from the video frames or images of the user. While first solution uses some of the pose landmarks to calculate the joint angles and estimate the exercise pose, this second solution uses a dataset of exercise pose images and extract the pose landmarks data into a csv file. The data in the csv file is then used to build a prediction model for the exercise poses, which is used to predict the exercise pose of the user and if the user is doing the "up" or "down" version of that exercise pose through the pose landmarks data of the user extracted from the images from the video frames captured by the webcam on the user's device.

Libraries Used:
- opencv-python
- mediapipe
- pandas
- scikit-learn

References Used:
- https://github.com/pereldegla/yoga_assistant
- https://www.kaggle.com/datasets/muhannadtuameh/exercise-recognition

For the references mentioned, we used the GitHub repository for a program that uses pose detection in machine learning to predict the user's yoga poses as a base reference for our code. We modified and refined the code into something that fits our needs better to predict the exercise poses better. We also tried to use the training dataset for Physical Exercise Recognition on kaggle.com that contained pose landmarks data extracted using MediaPipe Pose, but it did not work with our code as it was missing visibility values for the pose landmarks despite our attempts to modify the code to make the prediction work only using the xyz coordinates. However, it did give us a better idea on how the training dataset for the code should look like for the purposes we wanted to use it for. We also searched for more datasets on kaggle.com and other sources, but they were either in a different format than we needed, or were images not categorized into "up" and "down" poses which we needed. In the end, we ended up building our own images dataset by using Google Images, resulting in 20 images for each exercise pose (pushups up, pushups down, squats up, squats down), a total of 80 images.

Going back to the code, there are 2 different files:
- train.py (For extracting pose landmarks data from the images dataset)
- predict.py (For training the prediction model and predicting user pose)

Instructions on how to get started (Developed on a M1 MacBook Air using PyCharm):
1. Type "pip install -r requirements.txt" into the console to install required libraries.
2. Go into "train.py" and make sure that the path for the images dataset folder is correct.
3. Type "python3 train.py" into the console and the csv file for training data should be created.
4. Type "python3 predict.py" into the console and switch tabs to the opened webcam window.

That should be all for the instructions, but some changes might be needed for running this code depending on the device or operating system.