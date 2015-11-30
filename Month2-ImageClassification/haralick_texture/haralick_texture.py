# -- coding: utf-8 --

__author__ = 'amryfitra'

from sklearn.svm import LinearSVC
import mahotas
import cv2
import os
import numpy as np

from IPython import embed

# initialize the data matrix and the list of labels
print("[INFO] extracting features...")

train_dataset = []
train_dataset_label = []

for dirpath, folders, files in os.walk("training"):
    # load the image, convert it to grayscale, and extract the texture
    # name from the filename
    for file in files:
        if file.endswith(".png") or file.endswith(".bmp"):
            imagePath = os.path.join(dirpath, file)
            image = cv2.imread(imagePath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            texture = imagePath[imagePath.rfind("/") + 1:].split("_")[0]

            # extract Haralick texture features in 4 directions, then take the
            # mean of each direction
            haralick_features = mahotas.features.haralick(image).mean(axis=0)

            # update the data and labels
            train_dataset.append(haralick_features)
            train_dataset_label.append(texture)

# train the classifier
print("[INFO] training model...")
model = LinearSVC(C=10.0, random_state=42)
model.fit(train_dataset, train_dataset_label)
print("[INFO] classifying...")

for dirpath, folders, files in os.walk("test"):
    # load the image, convert it to grayscale, and extract Haralick
    # texture from the test image
    for file in files:
        if file.endswith(".png") or file.endswith(".bmp"):
            imagePath = os.path.join(dirpath, file)
            image = cv2.imread(imagePath)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            features = mahotas.features.haralick(gray).mean(axis=0)

            # classify the test image
            pred = model.predict(features)[0]
            cv2.putText(image, pred, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (0, 255, 0), 3)

            # show the output image
            cv2.imshow("Image", image)
            cv2.waitKey(0)


