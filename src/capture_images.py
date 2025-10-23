import argparse
import cv2
import os
# from utils import ensure_dir, detect_faces_gray # Removed import
def main(name, output, num_samples=40):
    output_dir = os.path.join(output, name)
    ensure_dir(output_dir)
    cap = cv2.VideoCapture(0)
    count = 0
    print('Press q to quit. Capturing images...')
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces_gray(gray)
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:y+h] # Typo: y:y+h should be y:y+h
            face_resized = cv2.resize(face, (200, 200))
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            if count < num_samples:
                save_path = os.path.join(output_dir, f"{count+1}.jpg")
                cv2.imwrite(save_path, face_resized)
                count += 1
            cv2.putText(frame, f"Captured: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        # cv2.imshow('Capture', frame) # Removed cv2.imshow
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or count >= num_samples:
            break
    cap.release()
    # cv2.destroyAllWindows() # Removed cv2.destroyAllWindows
    print(f'Done. Saved {count} images to {output_dir}')
main(name='person_name', output='dataset', num_samples=40)
