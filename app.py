import cv2
import os
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
labels = []
label_map = {}
label_id = 0

for root, dirs, files in os.walk("known_faces"):
    for subdir in dirs:
        path = os.path.join(root, subdir)
        for file in os.listdir(path):
            if file.lower().endswith(('png', 'jpg', 'jpeg')):
                img_path = os.path.join(path, file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                faces_rect = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)

                for (x, y, w, h) in faces_rect:
                    face = img[y:y+h, x:x+w]
                    faces.append(face)
                    if subdir not in label_map:
                        label_map[subdir] = label_id
                        label_id += 1
                    labels.append(label_map[subdir])

if len(faces) < 2:
    print("❌ Not enough data to train. Add more face images.")
    exit()

recognizer.train(faces, np.array(labels))
recognizer.save("recognizer.yml")

# Save label map
with open("labels.txt", "w") as f:
    for name, id_ in label_map.items():
        f.write(f"{id_}:{name}\n")

print("✅ Training complete. Model saved as recognizer.yml")
