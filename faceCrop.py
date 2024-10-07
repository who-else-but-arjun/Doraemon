import cv2
import os
from pathlib import Path

# Load the pre-trained Haar Cascade model for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def extract_faces(video_path):
    video_name = Path(video_path).stem
    output_dir = "faces"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    
    frame_count = 0
    face_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break 

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=8, 
            minSize=(50, 50) 
        )

        for (x, y, w, h) in faces:
            aspect_ratio = w / float(h)
            if 0.8 < aspect_ratio < 1.2: 
                face = frame[y:y+h, x:x+w]
                face_resized = cv2.resize(face, (224, 224))
                
                face_image_path = os.path.join(output_dir, f"face_{frame_count}_{face_count}.jpg")
                cv2.imwrite(face_image_path, face_resized) 
                print(f"Saved: {face_image_path}")
                face_count += 1

        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    return output_dir

if __name__ == "__main__":
    video_path = 'cold.mp4' 
    extract_faces(video_path)
