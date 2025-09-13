import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
from threading import Thread, Lock
import time
import mediapipe as mp


# Load Emotion Recognition Model
emotion_model = load_model('DETECTION_SYSTEM/EMOTION_DETECTION_SYSTEM/emotion_recognition_model.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load Haar Cascade for Face Detection
faceCascade = cv2.CascadeClassifier('DETECTION_SYSTEM/EMOTION_DETECTION_SYSTEM/haarcascade_frontalface_default.xml')

# Load LBPH Face Recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "/Users/trish/Documents/VS CODE/Projects/NTCC 2nd/FINAL PRODUCT/DETECTION_SYSTEM/FACE_RECOGINITION_SYSTEM/FACE_DETECTION/classifier.xml"))
recognizer.read(recognizer_path)

class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(1)
        self.video.set(cv2.CAP_PROP_FPS, 15)
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.running = True
        self.lock = Lock()
        self.frame = None
        self.emotion = "Neutral"
        self.fps_counter = 0
        self.prev_time = time.time()
        self.frame_count = 0

        # MediaPipe Face Detector
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detector = self.mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6)

        self.thread = Thread(target=self.update_frame, daemon=True)
        self.thread.start()

    def update_frame(self):
        while self.running:
            success, frame = self.video.read()
            if not success:
                continue

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_detector.process(image_rgb)

            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    h, w, _ = frame.shape
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    bw = int(bbox.width * w)
                    bh = int(bbox.height * h)

                    x, y = max(0, x), max(0, y)
                    bw, bh = min(w - x, bw), min(h - y, bh)

                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    name = self.get_identity(gray, x, y, bw, bh)

                    face_crop = gray[y:y+bh, x:x+bw]
                    if self.frame_count % 10 == 0 and face_crop.size != 0:
                        resized = cv2.resize(face_crop, (48, 48))
                        normalized = resized.astype('float32') / 255.0
                        reshaped = np.reshape(normalized, (1, 48, 48, 1))
                        prediction = emotion_model.predict(reshaped, verbose=0)
                        self.emotion = emotion_labels[np.argmax(prediction)]

                    label = f"{name} | {self.emotion}"
                    cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    break  # Process only first detected face for speed

            # FPS counter
            self.fps_counter += 1
            now = time.time()
            if now - self.prev_time >= 1.0:
                self.current_fps = self.fps_counter
                self.fps_counter = 0
                self.prev_time = now

            cv2.putText(frame, f"FPS: {getattr(self, 'current_fps', 0)}", (15, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            _, jpeg = cv2.imencode('.jpg', frame)
            with self.lock:
                self.frame = jpeg.tobytes()

            self.frame_count += 1
            time.sleep(0.01)

    def get_identity(self, gray_img, x, y, w, h):
        try:
            id, pred = recognizer.predict(gray_img[y:y+h, x:x+w])
            confidence = int(100 * (1 - pred / 300))
            if confidence > 70:
                if id == 1:
                    return "Trish"
                if id == 2:
                    return "Amitesh"
            return "UNKNOWN"
        except:
            return "UNKNOWN"

    def get_frame(self):
        with self.lock:
            return self.frame

    def __del__(self):
        self.running = False
        self.video.release()
