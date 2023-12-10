import cv2
import mediapipe as mp
import time


class FaceDetector:
    def __init__(self, min_detection_confidence=0.5, model_selection=0):
        """
        Initializes the FaceDetector object.

        Parameters:
        - min_detection_confidence: Minimum confidence threshold for face detection.
        - model_selection: Model selection for face detection (0 or 1).
        """
        self.min_detection_confidence = min_detection_confidence
        self.model_selection = model_selection

        self.mp_draw = mp.solutions.drawing_utils
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            self.min_detection_confidence, self.model_selection)

    def find_face(self, img, draw=True):
        """
        Finds faces in the given image.

        Parameters:
        - img: Input image.
        - draw: Boolean flag indicating whether to draw bounding boxes on the image.

        Returns:
        - img: Image with bounding boxes drawn (if draw is True).
        - b_box_s: List of face bounding boxes and scores.
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.face_detection.process(img_rgb)
        b_box_s = []

        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                b_box_class = detection.location_data.relative_bounding_box
                img_height, img_width, img_class = img.shape
                b_box = (
                    int(b_box_class.xmin * img_width),
                    int(b_box_class.ymin * img_height),
                    int(b_box_class.width * img_width),
                    int(b_box_class.height * img_height)
                )
                b_box_s.append([id, b_box, detection.score])
                if draw:
                    img = self.fansy_draw(img, b_box)

                    cv2.putText(img, f"{int(detection.score[0] * 100)} %",
                                (b_box[0], b_box[1] - 20),
                                cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

        return img, b_box_s

    def fansy_draw(self, img, b_box, l=30, t=5, rt=1):
        x, y, w, h = b_box
        x1, y1 = x + w, y + h

        cv2.rectangle(img, b_box, (255, 0, 255), rt)
        # Top Left x, y
        cv2.line(img, (x, y), (x + l, y), (255, 0, 255), t)
        cv2.line(img, (x, y), (x, y + l), (255, 0, 255), t)
        # Top Right x1, y
        cv2.line(img, (x1, y), (x1 - l, y), (255, 0, 255), t)
        cv2.line(img, (x1, y), (x1, y + l), (255, 0, 255), t)
        # Bottom Left x, y1
        cv2.line(img, (x, y1), (x + l, y1), (255, 0, 255), t)
        cv2.line(img, (x, y1), (x, y1 - l), (255, 0, 255), t)
        # Bottom Right x1, y1
        cv2.line(img, (x1, y1), (x1 - l, y1), (255, 0, 255), t)
        cv2.line(img, (x1, y1), (x1, y1 - l), (255, 0, 255), t)

        return img


def main():
    cap = cv2.VideoCapture('../Videos/6.mp4')
    previous_time = 0
    detector = FaceDetector()

    while True:
        success, img = cap.read()

        img, b_box_s = detector.find_face(img)
        print(b_box_s)

        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time

        cv2.putText(img, f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (0, 255, 0), 2)
        cv2.imshow('Image', img)

        cv2.waitKey(1)


if __name__ == '__main__':
    main()
