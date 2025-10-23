import argparse
import cv2
import os
import pickle
import numpy as np # Moved import outside of if block
# from utils import prepare_training_data, ensure_dir # Removed import
def main(data_dir, model_path):
    ensure_dir(os.path.dirname(model_path) or '.')
    faces, labels, label_map = prepare_training_data(data_dir)
    if len(faces) == 0:
        print('No faces found in dataset. Please capture images first.')
        return
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))
    recognizer.write(model_path)
    # Save label map
    with open(model_path + '.labels.pkl', 'wb') as f:
        pickle.dump(label_map, f)
    print(f'Trained model saved to {model_path}')
# Direct call to main with default values
main(data_dir='dataset', model_path='models/lbph_model.xml')
