import cv2
import PoseEstimationModule as pem
import time


cap = cv2.VideoCapture('PoseVideos/5.mp4')
previous_time = 0
detector = pem.PoseDetector()

while True:
    success, img = cap.read()
    img = detector.find_pose(img)
    lm_list = detector.find_position(img)
    print(lm_list)
    # if len(lm_list) != 0:
    #     print(lm_list[14])
    #     cv2.circle(img, (lm_list[14][1], lm_list[14][2]), 15, (0, 0, 255), cv2.FILLED)

    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time

    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)

    cv2.imshow('Image', img)
    cv2.waitKey(1)
