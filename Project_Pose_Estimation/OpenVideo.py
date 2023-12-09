import cv2  # OpenCV is used to work with the video stream and image processing.
import mediapipe as mp  # Mediapipe is used to detect and track key points on the hands.


cap = cv2.VideoCapture('PoseVideos/1.mp4')

while True:
    success, img = cap.read()

    cv2.imshow('Image', img)
    cv2.waitKey(1)
