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

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray image", gray)

(h, w) = img.shape[:2]

K = np.ones((3, 3), dtype=np.uint8) * (1/9)

one = np.uint8([1])
result = gray.copy()

for x in range(0, w):
    if cv2.add(x, one) < w and cv2.subtract(x, one) >= 0:
        for y in range(0, h):
            if cv2.add(y, one) < h and cv2.subtract(y, one) >= 0:
                """'''''''
                | a  b c |
                | d cp e |
                | f  g h |
                '''''''"""
                roi = np.uint8([
                    [gray[y-1, x - 1], gray[y-1, x], gray[y-1, x + 1]],
                    [gray[y, x - 1], gray[y, x], gray[y, x + 1]],
                    [gray[y+1, x - 1], gray[y+1, x], gray[y+1, x + 1]]
                ])
                res_M = (roi * K).sum(dtype=np.uint8)
                result[y, x] = (roi * K).sum()
                print("Point x = {}, y = {}".format(x, y))
                print("Before value: {}, After value: {}".format(gray[y, x], result[y,x]))
            else:
                print("y : {} hit boundary".format(y))
    else:
        print("x : {} hit boundary".format(x))

print("h,w of image = {},{}".format(h,w))
cv2.imshow("Result", result)

cv2.waitKey(0)

