import cv2  # OpenCV is used to work with the video stream and image processing.
import mediapipe as mp  # Mediapipe is used to detect and track key points on the hands.
import time


cap = cv2.VideoCapture('videos/7.mp4')
previous_time = 0

while True:
    success, img = cap.read()

    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time

    cv2.putText(img, str(int(fps)), (20, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                (0, 255, 0), 2)
    cv2.imshow('Image', img)

    cv2.waitKey(20)

