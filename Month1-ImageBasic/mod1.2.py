# -- coding: utf-8 --

__author__ = 'amryfitra'

import cv2
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="Image path")

args = vars(ap.parse_args())

img = cv2.imread(args["image"])

cv2.imshow("Original", img)

(h, w) = img.shape[:2]

(cX, cY) = (w/2, h/2)

top_left = img[0:cY, 0:cX]
top_right = img[0:cY, cX:w]
bot_left = img[cY:h, 0:cX]
bot_right = img[cY:h, cX:w]

cv2.imshow("Top left", top_left)
cv2.imshow("Top right", top_right)
cv2.imshow("Bot left", bot_left)
cv2.imshow("Bot right", bot_right)

img_cp = img.copy()
img_cp[0:cY, 0:cX] = (0, 255, 255)

cv2.imshow("Pixel Manipulation", img_cp)
cv2.waitKey(0)