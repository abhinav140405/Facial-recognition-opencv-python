import argparse
import cv2
import pickle
import os
# from utils import detect_faces_gray # Removed import
def main(model_path, data_dir):
    if not os.path.exists(model_path):
        print('Model not found. Please train model first.')
        return
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(model_path)
    with open(model_path + '.labels.pkl', 'rb') as f:
        label_map = pickle.load(f)
    cap = cv2.VideoCapture(0)
    print('Press q to quit.')
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detect_faces_gray(gray)
        for (x, y, w, h) in rects:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (200, 200))
            label, confidence = recognizer.predict(face)
            name = label_map.get(label, 'Unknown')
            text = f"{name} ({confidence:.1f})"
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        # cv2.imshow('Recognize', frame) # Removed cv2.imshow
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    cap.release()
main(model_path='models/lbph_model.xml', data_dir='dataset')
