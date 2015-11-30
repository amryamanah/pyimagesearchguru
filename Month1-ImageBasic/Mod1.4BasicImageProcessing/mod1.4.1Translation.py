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

# NOTE: Translating (shifting) an image is given by a NumPy matrix in
# the form:
#	[[1, 0, shiftX], [0, 1, shiftY]]
# You simply need to specify how many pixels you want to shift the image
# in the X and Y direction -- let's translate the image 25 pixels to the
# right and 50 pixels down

M_down_right = np.float32([
    [1, 0, 25],
    [0, 1, 50]
])

shifted_down_right = cv2.warpAffine(img, M_down_right, (img.shape[1], img.shape[0]))
cv2.imshow("shifted Down and Right", shifted_down_right)

# now, let's shift the image 50 pixels to the left and 90 pixels up, we
# accomplish this using negative values
M_up_left = np.float32([
    [1, 0, -50],
    [0, 1, -90]
])

shifted_up_left = cv2.warpAffine(img, M_up_left, (img.shape[1], img.shape[0]))
cv2.imshow("Shifted Up and Left", shifted_up_left)

cv2.waitKey(0)

