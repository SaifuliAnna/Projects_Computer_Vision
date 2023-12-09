import cv2  # OpenCV is used to work with the video stream and image processing.
import mediapipe as mp  # Mediapipe is used to detect and track key points on the hands.
import time


class PoseDetector:
    def __init__(self, static_mode=False, model_complexity=1, smooth_landmarks=True, enable_segmentation=False,
                 smooth_segmentation=True, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.static_mode = static_mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mp_draw = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(self.static_mode, self.model_complexity, self.smooth_landmarks,
                                      self.enable_segmentation, self.smooth_segmentation, self.min_detection_confidence,
                                      self.min_tracking_confidence)

    def find_pose(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img_rgb)
        if self.results.pose_landmarks:
            if draw:
                self.mp_draw.draw_landmarks(img, self.results.pose_landmarks,
                                            self.mp_pose.POSE_CONNECTIONS)

        return img

    def find_position(self, img, draw=True):
        lm_list = []
        for id, lm in enumerate(self.results.pose_landmarks.landmark):
            height, weight, chanel = img.shape
            chanel_x, chanel_y = int(lm.x * weight), int(lm.y * height)
            lm_list.append([id, chanel_x, chanel_y])
            if draw:
                cv2.circle(img, (chanel_x, chanel_y), 5, (255, 0, 0), cv2.FILLED)

        return lm_list


def main():
    cap = cv2.VideoCapture('PoseVideos/5.mp4')
    previous_time = 0
    detector = PoseDetector()

    while True:
        success, img = cap.read()
        img = detector.find_pose(img)
        lm_list = detector.find_position(img)
        # print(lm_list)

        if len(lm_list) != 0:
            print(lm_list[14])
            cv2.circle(img, (lm_list[14][1], lm_list[14][2]), 15, (0, 0, 255), cv2.FILLED)

        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 0), 3)

        cv2.imshow('Image', img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
