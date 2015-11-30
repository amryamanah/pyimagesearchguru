# -- coding: utf-8 --

__author__ = 'amryfitra'

import numpy as np
import argparse
import cv2

import imutils

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to image")
args = vars(ap.parse_args())

img = cv2.imread(args["image"])
cv2.imshow("Original image", img)

(h, w) = img.shape[:2]
# we need to keep in mind aspect ratio so the image does not look skewed
# or distorted -- therefore, we calculate the ratio of the new image to
# the old image. Let's make our new image have a width of 150 pixels

r_w = 150.0 / w
dim_w = (150, int(h*r_w))

# perform the actual resizing of the image
resized_w = cv2.resize(img, dim_w, interpolation=cv2.INTER_AREA)
cv2.imshow("Resized based on width", resized_w)


# what if we wanted to adjust the height of the image? We can apply
# the same concept, again keeping in mind the aspect ratio, but instead
# calculating the ratio based on height -- let's make the height of the
# resized image 50 pixels

r_h = 50.0 / h
dim_h = (int(w * r_h), 50)
resized_h = cv2.resize(img, dim_h, interpolation=cv2.INTER_AREA)
cv2.imshow("Resized based on height", resized_h)


# construct the list of interpolation methods
methods = [
    ("cv2.INTER_NEAREST", cv2.INTER_NEAREST),
    ("cv2.INTER_LINEAR", cv2.INTER_LINEAR),
    ("cv2.INTER_AREA", cv2.INTER_AREA),
    ("cv2.INTER_CUBIC", cv2.INTER_CUBIC),
    ("cv2.INTER_LANCZOS4", cv2.INTER_LANCZOS4),
]

for (name, method) in methods:
    result = imutils.resize(img, width=w * 3, inter=method)
    cv2.imshow(name, result)
    cv2.waitKey(0)
