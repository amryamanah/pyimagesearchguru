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

# flip the image horizontally
h_flip = cv2.flip(img, 1)
cv2.imshow("Flipped Horizontally", h_flip)

# flip the image vertically
v_flip = cv2.flip(img, 0)
cv2.imshow("Flipped Vertically", v_flip)

# flip the image along both axes
vh_flip = cv2.flip(img, -1)
cv2.imshow("Flipped Horizontal & Vertical", vh_flip)


cv2.waitKey(0)