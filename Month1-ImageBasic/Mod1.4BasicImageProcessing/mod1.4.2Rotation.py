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
(cX, cY) = (w/2, h/2)

Mr_45 = cv2.getRotationMatrix2D((cX, cY), 45, 1.0)
r_45 = cv2.warpAffine(img, Mr_45, (w, h))
cv2.imshow("Rotated 45 degree", r_45)

Mr_min90 = cv2.getRotationMatrix2D((cX, cY), -90, 1.0)
r_min90 = cv2.warpAffine(img, Mr_min90, (w, h))
cv2.imshow("Rotated -90 degree", r_min90)

cv2.waitKey(0)
