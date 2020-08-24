import os
import cv2
import pickle
import numpy as np

from PIL import Image

#  Gets all the images in the folder train.
base_dir = os.path.dirname(os.path.abspath(__file__))

img_dir = os.path.join(base_dir, "train")
recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier(os.path.join(base_dir, r"cascades\haarcascade_frontalface_default.xml"))

current_id = 0
label_ids = {}
y_label = []
x_train = []

for root, dirs, files in os.walk(img_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()

            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]

            pil_img = Image.open(path).convert("L")
            size = (550, 550)
            final_img = pil_img.resize(size, Image.ANTIALIAS)
            img_array = np.array(final_img, "uint8")
            
            faces = face_cascade.detectMultiScale(img_array, 1.3, 5)

            for (x, y, w, h) in faces:
                roi = img_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_label.append(id_)

with open(os.path.join(base_dir, "labels.pickle"), "wb") as data_file:
    pickle.dump(label_ids, data_file)

recognizer.train(x_train, np.array(y_label))
recognizer.save(os.path.join(base_dir, "trainner.yml"))

print("[*] Completed")