# -- coding: utf-8 --

import cv2
from IPython import embed
import numpy as np
import imutils


def approx_simple():
    image = cv2.imread("../img/circles_and_squares.png")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    (_, cnts, _) = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.01 * peri, True)
        # if the approximated contour has 4 vertices, then we are examining
        # a rectangle
        if len(approx) == 4:
            cv2.drawContours(image, [c], -1, (0, 255, 255), 2)
            (x, y, w, h) = cv2.boundingRect(approx)
            cv2.putText(image, "Rectangle", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    cv2.imshow("image", image)


def approx_realworld():
    image = cv2.imread("../img/receipt.png")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edged = imutils.auto_canny(gray, 3.33)

    cv2.imshow("Original", image)
    cv2.imshow("Edge", edged)

    (_, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:7]

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.01 * peri, True)

        print("original: {}, approx: {}".format(len(c), len(approx)))

        if len(approx) == 4:
            cv2.drawContours(image, [approx], -1, (0, 255, 0), 3)

    cv2.imshow("Image", image)


if __name__ == '__main__':
    #approx_simple()
    approx_realworld()
    cv2.waitKey(0)