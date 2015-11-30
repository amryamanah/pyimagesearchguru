# -- coding: utf-8 --

__author__ = 'amryfitra'

import numpy as np
import argparse
import cv2

import imutils

img = cv2.imread("img/florida_trip.png")
cv2.imshow("Original image", img)

# cropping an image is accomplished using simple NumPy array slices --
# let's crop the face from the image
face = img[85:250, 85:220]
cv2.imshow("Face", face)
cv2.waitKey(0)

body = img[90:450, 0:290]
cv2.imshow("Full body", body)
cv2.waitKey(0)