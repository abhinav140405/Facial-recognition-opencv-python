import os
import cv2
import numpy as np
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
def detect_faces_gray(image):
    # expects a grayscale image
    faces = FACE_CASCADE.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    return faces
def prepare_training_data(data_dir):
    """Walk dataset directory and return (faces, labels, label_map)
    Assumes structure: data_dir/person_name/1.jpg ..."""
    faces = []
    labels = []
    label_map = {}
    label_counter = 0
    for person_name in sorted(os.listdir(data_dir)):
        person_path = os.path.join(data_dir, person_name)
        if not os.path.isdir(person_path):
            continue
        label_map[label_counter] = person_name
        for filename in os.listdir(person_path):
            filepath = os.path.join(person_path, filename)
            img = cv2.imread(filepath)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            rects = detect_faces_gray(gray)
            if len(rects) == 0:
                continue
            (x, y, w, h) = rects[0]
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (200, 200))
            faces.append(face)
            labels.append(label_counter)
        label_counter += 1
    return faces, labels, label_map
