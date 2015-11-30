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

# Masking allows us to focus only on parts of an image that interest us.
# A mask is the same size as our image, but has only two pixel values,
# 0 and 255. Pixels with a value of 0 are ignored in the orignal image,
# and mask pixels with a value of 255 are allowed to be kept. For example,
# let's construct a rectangular mask that displays only the person in
# the image

mask = np.zeros(img.shape[:2], dtype=np.uint8)
cv2.rectangle(mask, (0, 90), (290, 450), 255, -1)
cv2.imshow("Mask", mask)

masked = cv2.bitwise_and(img, img, mask=mask)
cv2.imshow("Mask applied to image", masked)

circle_mask = np.zeros(img.shape[:2], dtype=np.uint8)
cv2.circle(circle_mask, (145, 200), 100, 255, -1)
masked_circle = cv2.bitwise_and(img, img, mask=circle_mask)
cv2.imshow("Circle mask applied to image", masked_circle)

cv2.waitKey(0)