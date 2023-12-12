import cv2  # OpenCV is used to work with the video stream and image processing.
import mediapipe as mp  # Mediapipe is used to detect and track key points on the hands.
import time


class FaceMeshDetector:
    def __init__(self, static_image_mode=False, max_num_faces=4,
                 refine_landmarks=False, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):

        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mp_draw = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(self.static_image_mode, self.max_num_faces,
                                                    self.refine_landmarks, self.min_detection_confidence,
                                                    self.min_tracking_confidence)
        self.draw_spec = self.mp_draw.DrawingSpec(thickness=1, circle_radius=1)

    def fiind_mesh_face(self, img, draw=True):
        self.img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.face_mesh.process(self.img_rgb)
        faces = []
        if self.results.multi_face_landmarks:
            for face_lms in self.results.multi_face_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, face_lms, self.mp_face_mesh.FACEMESH_CONTOURS,
                                           self.draw_spec, self.draw_spec)
                face = []
                for id, lm in enumerate(face_lms.landmark):
                    # print(lm)
                    img_height, img_width, img_chanel = img.shape
                    x, y = int(lm.x * img_width), int(lm.y * img_height)
                    # cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN, 0.5,
                    #             (0, 255, 0), 1)
                    # print(id, x, y)
                    # face.append([id, x, y])
                    face.append([x, y])
                faces.append(face)

        return img, faces


def main():
    cap = cv2.VideoCapture(0)
    previous_time = 0
    detector = FaceMeshDetector()
    while True:
        success, img = cap.read()
        img, faces = detector.fiind_mesh_face(img)
        if len(faces) != 0:
            # print(len(faces[0]))
            print(faces[0])

        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time

        cv2.putText(img, f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (0, 255, 0), 3)
        cv2.imshow('Image', img)

        cv2.waitKey(20)


if __name__ == "__main__":
    main()
