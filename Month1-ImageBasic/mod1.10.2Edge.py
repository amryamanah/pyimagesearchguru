# -- coding: utf-8 --

import cv2
import argparse
import imutils
import numpy as np


def canny_edge(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    cv2.imshow("Gray", gray)
    cv2.imshow("Blurred", blurred)

    # compute a "wide", "mid-range", and "tight" threshold for the edges
    wide = cv2.Canny(blurred, 10, 200)
    mid = cv2.Canny(blurred, 30, 150)
    tight = cv2.Canny(blurred, 245, 250)
    auto_range = imutils.auto_canny(blurred)

    cv2.imshow("Wide Edge Map", wide)
    cv2.imshow("Mid Edge Map", mid)
    cv2.imshow("Tight Edge Map", tight)
    cv2.imshow("Auto canny", auto_range)

if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to image")

    args = vars(ap.parse_args())

    img = cv2.imread(args["image"])
    img = cv2.imread(args["image"])

    canny_edge(img)
    cv2.waitKey(0)
