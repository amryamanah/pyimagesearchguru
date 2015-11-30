# -- coding: utf-8 --

__author__ = 'amryfitra'

import argparse
import cv2
import imutils


def show_rgb(image):
    for (name, chan) in zip(("B", "G", "R"), cv2.split(image)):
        print("{}, max: {}, min: {}".format(name, chan.max(), chan.min()))
        cv2.imshow(name, chan)

    cv2.waitKey(0)


def show_hsv(image):
    # Hue: Which “pure” color we are examining.
    # For example, all shadows and tones of the color “red” will have the same Hue.

    # Saturation: How “white” the color is. A fully saturated color would be “pure,” as in “pure red.”
    # And a color with zero saturation would be pure white.

    # Value: The Value allows us to control the lightness of our color. A Value of zero would indicate pure black,
    # whereas increasing the value would produce lighter colors.

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cv2.imshow("HSV", hsv)
    for (name, chan) in zip(("H", "S", "V"), cv2.split(hsv)):
        print("{}, max: {}, min: {}".format(name, chan.max(), chan.min()))
        cv2.imshow(name, chan)
    cv2.waitKey(0)


def show_lab(image):
    # L-channel: The “lightness” of the pixel. This value goes up and down the vertical axis, white to black,
    # with neutral grays at the center of the axis.

    # a-channel: Originates from the center of the L-channel and
    # defines pure green on one end of the spectrum and pure red on the other.

    # b-channel: Also originates from the center of the L-channel, but is perpendicular to the a-channel.
    # The b-channel defines pure blue at one of the spectrum and pure yellow at the other.

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    cv2.imshow("L*a*b*", lab)
    for (name, chan) in zip(("L*", "a*", "b*"), cv2.split(lab)):
        print("{}, max: {}, min: {}".format(name, chan.max(), chan.min()))
        cv2.imshow(name, chan)
    cv2.waitKey(0)


def show_grayscale(image):
    # A grayscale representation of an image throws away a color information of an image
    # Y = 0.299 * R + 0.587 * G + 0.114 * B
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Gray", gray)
    cv2.waitKey(0)


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to image")
    args = vars(ap.parse_args())

    img = cv2.imread(args["image"])
    img = imutils.resize(img,width=500)
    cv2.imshow("Original", img)
    # show_rgb(img)
    show_hsv(img)
    show_lab(img)

    cv2.destroyAllWindows()