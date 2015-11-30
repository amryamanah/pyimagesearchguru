# -- coding: utf-8 --

__author__ = 'amryfitra'

import cv2
import argparse
import imutils
import numpy as np


def sobel_gradient(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Gray", gray)

    # compute gradients along the X and Y axis, respectively
    gX = cv2.Sobel(gray, ddepth=cv2.CV_64F, dx=1, dy=0)
    gY = cv2.Sobel(gray, ddepth=cv2.CV_64F, dx=0, dy=1)

    # the `gX` and `gY` images are now of the floating point data type,# so we need to take care to convert them back a to unsigned 8-bit
    # integer representation so other OpenCV functions can utilize them
    gX = cv2.convertScaleAbs(gX)
    gY = cv2.convertScaleAbs(gY)

    # combine the sobel X and Y representations into a single image
    sobel_combined = cv2.addWeighted(gX, 0.5, gY, 0.5, 0)

    cv2.imshow("Gradient X", gX)
    cv2.imshow("Gradient Y", gY)
    cv2.imshow("Gradient XY", sobel_combined)


def cal_grad_mag_orientation(image, lower_angle, upper_angle):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Gray", gray)

    # compute gradients along the X and Y axis, respectively
    gX = cv2.Sobel(gray, ddepth=cv2.CV_64F, dx=1, dy=0)
    gY = cv2.Sobel(gray, ddepth=cv2.CV_64F, dx=0, dy=1)

    # calculate magnitude and gradient
    magnitude = np.sqrt((gX ** 2) + (gY ** 2))
    orientation = np.arctan2(gY, gX) * (180/np.pi) % 180

    # find all pixels that are within the upper and low angle boundaries
    idxs = np.where(np.logical_and(orientation >= lower_angle, orientation <= upper_angle), orientation, -1)

    mask = np.zeros(gray.shape, dtype=np.uint8)
    mask[idxs > -1] = 255

    # show the images
    cv2.imshow("Mask", mask)

if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to image")
    ap.add_argument("-l", "--lower-angle", type=float, default=175.0,
        help="Lower orientation angle")
    ap.add_argument("-u", "--upper-angle", type=float, default=180.0,
        help="Upper orientation angle")
    args = vars(ap.parse_args())

    img = cv2.imread(args["image"])
    img = imutils.resize(img,width=500)

    sobel_gradient(img)
    cal_grad_mag_orientation(img, args["lower_angle"], args["upper_angle"])

    cv2.waitKey(0)