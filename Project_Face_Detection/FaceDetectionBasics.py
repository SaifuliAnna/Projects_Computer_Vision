import cv2  # OpenCV is used to work with the video stream and image processing.
import mediapipe as mp  # Mediapipe is used to detect and track key points on the hands.
import time


cap = cv2.VideoCapture('videos/5.mp4')
previous_time = 0

mp_face_detection = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection()
# fae_detection = mp_face_detection.FaceDetection(0.75)

while True:
    success, img = cap.read()

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detection.process(img_rgb)
    print(results)

    if results.detections:
        for id, detection in enumerate(results.detections):
            # mp_draw.draw_detection(img, detection)
            # print(id, detection)
            # print(detection.score)
            print(detection.location_data.relative_bounding_box)
            b_box_class = detection.location_data.relative_bounding_box
            img_height, img_weight, img_chanel = img.shape
            b_box = int(b_box_class.xmin * img_weight), int(b_box_class.ymin * img_height), \
                    int(b_box_class.width * img_weight), int(b_box_class.height * img_height)

            cv2.rectangle(img, b_box, (255, 0, 255), 2)
            cv2.putText(img, f"{int(detection.score[0] * 100)} %", (b_box[0], b_box[1] - 20),
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time

    cv2.putText(img, f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                (0, 255, 0), 2)
    cv2.imshow('Image', img)

    cv2.waitKey(20)

