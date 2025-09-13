import os
import cv2
from PIL import Image
import numpy as np

def train_classifier(data_dir):
    path = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    
    faces = []
    ids = []
    
    for image in path:
        if image.endswith('.DS_Store') or image.startswith('.'):
            continue  # skip hidden/system files

        try:
            img = Image.open(image).convert('L')
            imageNp = np.array(img, 'uint8')
            id = int(os.path.split(image)[1].split(".")[1])
            
            faces.append(imageNp)
            ids.append(id)
        except Exception as e:
            print(f"Skipping file {image}: {e}")
            continue
        
    if len(faces) == 0:
        print("No valid images found. Training aborted.")
        return

    ids = np.array(ids)

    # Train and save classifier
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)
    clf.write("DETECTION_SYSTEM/FACE_RECOGINITION_SYSTEM/FACE_DETECTION/classifier.xml")
    print("Training complete. Model saved.")

# Run
train_classifier("DETECTION_SYSTEM/FACE_RECOGINITION_SYSTEM/data")
