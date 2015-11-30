# -- coding: utf-8 --

__author__ = 'amryfitra'

import argparse
import cv2
import imutils


def average_blur(image):
    # as the size of the kernel increases, so will the amount in which the image is blurred
    kernel_sizes = [(3, 3), (9, 9), (15, 15)]
    for kernel_size in kernel_sizes:
        blurred = cv2.blur(image, kernel_size)
        cv2.imshow("Blurred: ({},{})".format(kernel_size[0], kernel_size[1]), blurred)
    cv2.waitKey(0)


def gaussian_blur(image):
    # The end result is that our image is less blurred, but more naturally blurred, than using the average method
    # Furthermore, based on this weighting we’ll be able to preserve more of the edges in our image as compared to average smoothing.
    kernel_sizes = [(3, 3), (9, 9), (15, 15)]
    for kernel_size in kernel_sizes:
        blurred = cv2.GaussianBlur(image, kernel_size, 0)
        cv2.imshow("Gaussian Blur: ({},{})".format(kernel_size[0], kernel_size[1]), blurred)
    cv2.waitKey(0)


def median_blur(image):
    # Traditionally, the median blur method has been most effective when removing salt-and-pepper noise
    # the kernel size for the median must be square
    # Furthermore (unlike the averaging method),
    # instead of replacing the central pixel with the average of the neighborhood,
    # we instead replace the central pixel with the median of the neighborhood.
    # Notice that we are no longer creating a “motion blur” effect like in averaging and Gaussian blurring —
    # instead, we are removing substantially more detail and noise.
    for kernel_size in (3, 9, 15):
        blurred = cv2.medianBlur(image, kernel_size)
        cv2.imshow("Median Blur: ({},{})".format(kernel_size, kernel_size), blurred)
    cv2.waitKey(0)


def bilateral_blur(image):
    # In order to reduce noise while still maintaining edges, we can use bilateral blurring.
    # Bilateral blurring accomplishes this by introducing two Gaussian distributions
    # The first Gaussian function only considers spatial neighbors.
    # That is, pixels that appear close together in the (x, y)-coordinate space of the image.
    # The second Gaussian then models the pixel intensity of the neighborhood,
    # ensuring that only pixels with similar intensity are included in the actual computation of the blur.
    # Intuitively, this makes sense. If pixels in the same (small) neighborhood have a similar pixel value,
    # then they likely represent the same object. But if two pixels in the same neighborhood have contrasting values,
    # then we could be examining the edge or boundary of an object — and we would like to preserve this edge.
    # Overall, this method is able to preserve edges of an image, while still reducing noise.
    # The largest downside to this method is that it is considerably slower than its counterparts.

    # The third argument is our color standard deviation, denoted as sigma_{color}.
    # A larger value for \sigma_{color} means
    # that more colors in the neighborhood will be considered when computing the blur.
    # If we let \sigma_{color} get too large in respect to the diameter,
    # then we essentially have broken the assumption  of bilateral filtering —
    # that only pixels of similar color should contribute significantly to the blur.

    # Finally, we need to supply the space standard deviation, which we call \sigma_{space}.
    # A larger value of \sigma_{space} means
    # that pixels farther out from the central pixel diameter will influence the blurring calculation.

    params = [(11, 21, 7), (11, 41, 21), (11, 61, 39)]
    for (diameter, sigma_color, sigma_space) in params:
        blurred = cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)
        cv2.imshow("Bilateral Blur w/ d: {}, sigma_col: {}, sigma_space: {}".format(diameter, sigma_color, sigma_space), blurred)
    cv2.waitKey(0)


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to image")
    args = vars(ap.parse_args())

    img = cv2.imread(args["image"])
    img = imutils.resize(img,width=500)
    #####
    # FOR CATTLE median blur with 9 x 9 kernel is good
    # cv2.MORPH_RECT  or cv2.MORPH_CROSS
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    (h, s, v) = cv2.split(hsv)

    img = h

    cv2.imshow("Original", img)

    # average_blur(img)
    # gaussian_blur(img)
    median_blur(img)
    bilateral_blur(img)
