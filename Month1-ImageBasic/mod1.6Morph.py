# -- coding: utf-8 --

__author__ = 'amryfitra'

import numpy as np
import argparse
import cv2
from IPython import embed

import imutils


def erotion(gray):

    cv2.imshow("Gray image", gray)
    for i in range(0,3):
        eroded = cv2.erode(gray, None, iterations=i+1)
        cv2.imshow("Eroded {} times".format(i + 1), eroded)
        cv2.waitKey(0)
    cv2.destroyAllWindows()


def dilation(gray):

    cv2.imshow("Gray image", gray)
    for i in range(0,3):
        dilated = cv2.dilate(gray, None, iterations=i+1)
        cv2.imshow("Dilated {} times".format(i + 1), dilated)
        cv2.waitKey(0)
    cv2.destroyAllWindows()


def opening(gray, struct_element=cv2.MORPH_RECT):
    # input -> erosion -> dilation
    # You can choose the shape of structuring element by
    # cv2.MORPH_RECT, cv2.MORPH_CROSS, cv2.MORP_ELLIPSE

    cv2.imshow("Gray image", gray)
    kernel_sizes = [(3, 3), (5, 5), (7, 7)]

    if struct_element is cv2.MORPH_RECT:
        struct_element_name = "cv2.MORPH_RECT"
    elif struct_element is cv2.MORPH_CROSS:
        struct_element_name = "cv2.MORPH_CROSS"
    elif struct_element is cv2.MORPH_ELLIPSE:
        struct_element_name = "cv2.MORPH_ELLIPSE"

    for kernel_size in kernel_sizes:
        kernel = cv2.getStructuringElement(struct_element, kernel_size)
        print(kernel)
        opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        cv2.imshow("Opening: ({}, {}) ({})".format(kernel_size[0], kernel_size[1], struct_element_name), opened)
        cv2.waitKey(0)


def closing(gray, struct_element=cv2.MORPH_RECT):
    # input -> dilation -> erosion
    # You can choose the shape of structuring element by
    # cv2.MORPH_RECT, cv2.MORPH_CROSS, cv2.MORP_ELLIPSE

    cv2.imshow("Gray image", gray)
    kernel_sizes = [(3, 3), (5, 5), (7, 7)]

    if struct_element is cv2.MORPH_RECT:
        struct_element_name = "cv2.MORPH_RECT"
    elif struct_element is cv2.MORPH_CROSS:
        struct_element_name = "cv2.MORPH_CROSS"
    elif struct_element is cv2.MORPH_ELLIPSE:
        struct_element_name = "cv2.MORPH_ELLIPSE"

    for kernel_size in kernel_sizes:
        kernel = cv2.getStructuringElement(struct_element, kernel_size)
        print(kernel)
        closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        cv2.imshow("Closing: ({}, {}) ({})".format(kernel_size[0], kernel_size[1], struct_element_name), closed)
        cv2.waitKey(0)


def morphological_gradient(gray, struct_element=cv2.MORPH_RECT):
    # difference between erosion and dilation
    # You can choose the shape of structuring element by
    # cv2.MORPH_RECT, cv2.MORPH_CROSS, cv2.MORP_ELLIPSE

    cv2.imshow("Gray image", gray)
    kernel_sizes = [(3, 3), (5, 5), (7, 7)]

    if struct_element is cv2.MORPH_RECT:
        struct_element_name = "cv2.MORPH_RECT"
    elif struct_element is cv2.MORPH_CROSS:
        struct_element_name = "cv2.MORPH_CROSS"
    elif struct_element is cv2.MORPH_ELLIPSE:
        struct_element_name = "cv2.MORPH_ELLIPSE"

    for kernel_size in kernel_sizes:
        kernel = cv2.getStructuringElement(struct_element, kernel_size)
        print(kernel)
        result = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
        cv2.imshow("Morph Gradient: ({}, {}) ({})".format(kernel_size[0], kernel_size[1], struct_element_name), result)
        cv2.waitKey(0)


def white_hat(gray, struct_element=cv2.MORPH_RECT):
    # or top hat
    # difference between original image and the opening
    # A top hat operation is used to reveal light regions of an image on dark backgrounds.
    # You can choose the shape of structuring element by
    # cv2.MORPH_RECT, cv2.MORPH_CROSS, cv2.MORP_ELLIPSE

    cv2.imshow("Gray image", gray)

    if struct_element is cv2.MORPH_RECT:
        struct_element_name = "cv2.MORPH_RECT"
    elif struct_element is cv2.MORPH_CROSS:
        struct_element_name = "cv2.MORPH_CROSS"
    elif struct_element is cv2.MORPH_ELLIPSE:
        struct_element_name = "cv2.MORPH_ELLIPSE"

    # similarly, a tophat (also called a "whitehat") operation will enable
    # us to find light regions on a dark background

    struct_kernel = cv2.getStructuringElement(struct_element, (13, 5))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, struct_kernel)
    cv2.imshow("Top hat", tophat)
    cv2.waitKey(0)


def black_hat(gray, struct_element=cv2.MORPH_RECT):
    # or top hat
    # difference between the closing of the input image and the input image itself
    # A top hat operation is used to reveal dark regions of an image on light backgrounds.
    # You can choose the shape of structuring element by
    # cv2.MORPH_RECT, cv2.MORPH_CROSS, cv2.MORP_ELLIPSE

    cv2.imshow("Gray image", gray)

    if struct_element is cv2.MORPH_RECT:
        struct_element_name = "cv2.MORPH_RECT"
    elif struct_element is cv2.MORPH_CROSS:
        struct_element_name = "cv2.MORPH_CROSS"
    elif struct_element is cv2.MORPH_ELLIPSE:
        struct_element_name = "cv2.MORPH_ELLIPSE"

   # construct a rectangular kernel and apply a blackhat operation which
    # enables us to find dark regions on a light background
    struct_kernel = cv2.getStructuringElement(struct_element, (13, 5))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, struct_kernel)
    cv2.imshow("Black hat", blackhat)
    cv2.waitKey(0)


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to image")
    args = vars(ap.parse_args())

    img = cv2.imread(args["image"])
    img = imutils.resize(img, 500)
    cv2.imshow("Original", img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #####
    # FOR CATTLE opening is good in hue channel (3,3) size structuring element
    # cv2.MORPH_RECT  or cv2.MORPH_CROSS
    # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # (h, s, v) = cv2.split(hsv)
    #
    # gray = h

    erotion(img.copy())
    dilation(img.copy())
    opening(img.copy())
    # opening(img.copy(), cv2.MORPH_CROSS)
    # opening(img.copy(), cv2.MORPH_ELLIPSE)
    # closing(img.copy())
    # closing(img.copy(), cv2.MORPH_CROSS)
    # closing(img.copy(), cv2.MORPH_ELLIPSE)
    # morphological_gradient(img.copy())
    # morphological_gradient(img.copy(), cv2.MORPH_CROSS)
    # morphological_gradient(img.copy(), cv2.MORPH_ELLIPSE)
    # white_hat(gray)
    # black_hat(gray)
    cv2.destroyAllWindows()



