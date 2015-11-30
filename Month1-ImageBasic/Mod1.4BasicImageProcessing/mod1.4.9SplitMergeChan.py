# -- coding: utf-8 --

__author__ = 'amryfitra'

import numpy as np
import argparse
import cv2
from IPython import embed

import imutils

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to image")
args = vars(ap.parse_args())

img = cv2.imread(args["image"])
cv2.imshow("Original image", img)

(b, g, r) = cv2.split(img)

cv2.imshow("Red", r)
cv2.imshow("Green", g)
cv2.imshow("Blue", b)

zeros = np.zeros(img.shape[:2], dtype=np.uint8)

red_only = cv2.merge([zeros, zeros, r])
green_only = cv2.merge([zeros, g, zeros])
blue_only = cv2.merge([b, zeros, zeros])

cv2.imshow("Red Only", red_only)
cv2.imshow("Green Only", green_only)
cv2.imshow("Blue Only", blue_only)


cv2.waitKey(0)