import cv2
import numpy as np
import time
import PoseModule as pes


cap = cv2.VideoCapture("../AI_Trainer_Project/curls.mp4")
detector = pes.PoseDetector()
count = 0
dir = 0
previous_time = 0

while True:
    success, img = cap.read()
    img = cv2.resize(img, (1280, 720))
    # img = cv2.imread("../AI_Trainer_Project/pose.jpg")
    img = detector.find_pose(img, False)
    lm_list = detector.find_position(img, False)
    # print(lm_list)

    if len(lm_list) != 0:
        # # right arm
        # detector.find_angle(img, 12, 14, 16)
        # left arm
        angle = detector.find_angle(img, 11, 13, 15)
        per = np.interp(angle, (192, 282), (0, 100))
        bar = np.interp(angle, (200, 290), (650, 100))
        print(angle, per)

        # check for the dumbbell curl
        color = (255, 0, 255)
        if per == 100:
            color = (0, 255, 0)
            if dir == 0:
                count += 0.5
                dir = 1
        if per == 0:
            color = (0, 255, 0)
            if dir == 1:
                count += 0.5
                dir = 0
        print(count)

        # Draw Bar
        cv2.rectangle(img, (1250, 100), (1280, 650), color, 3)
        cv2.rectangle(img, (1250, int(bar)), (1280, 650), color, cv2.FILLED)
        cv2.putText(img, f"{int(per)} %", (1175, 75), cv2.FONT_HERSHEY_PLAIN, 2,
                    color, 4)

        # Draw Curl Count
        cv2.rectangle(img, (0, 550), (150, 720), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(int(count)), (25, 685), cv2.FONT_HERSHEY_PLAIN, 10,
                    (255, 0, 255), 15)

    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time

    cv2.putText(img, str(int(fps)), (50, 100), cv2.FONT_HERSHEY_PLAIN, 5,
                (255, 0, 255), 5)

    cv2.imshow('Image', img)
    cv2.waitKey(1)
