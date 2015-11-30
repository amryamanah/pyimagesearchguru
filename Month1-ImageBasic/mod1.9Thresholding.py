# -- coding: utf-8 --

from skimage.filters import threshold_adaptive
import argparse
import cv2
import numpy as np
import imutils
from IPython import embed


def simple_thresholding(image, t_value):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    cv2.imshow("Gray", gray)
    cv2.imshow("Blurred", blurred)

    # apply basic thresholding --
    # first parameter is image
    # second parameter is threshold value
    # third is fall over value
    # in case cv2.THRESH_BINARY_INV if pixel > threshold it = 0 and pixel < threshold = thirdparameter value
    # in case cv2.THRESH_BINARY if pixel > threshold it = thirdparameter value and pixel < threshold = 0

    (T, thresh_inv) = cv2.threshold(blurred, t_value, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow("Thresh binary inverse", thresh_inv)

    # using normal thresholding (rather than inverse thresholding),
    # we can change the last argument in the function to make the coins
    # black rather than white.
    (T, thresh) = cv2.threshold(blurred, t_value, 255, cv2.THRESH_BINARY)
    cv2.imshow("Thresh binary", thresh)

    # finally, we can visualize only the masked regions in the image
    cv2.imshow("Output", cv2.bitwise_and(image, image, mask=thresh_inv))


def otsu_threshold(image):
    # Otsu’s method assumes that our image contains two classes of pixels: the background and the foreground.
    # Furthermore, Otsu’s method makes the assumption that the grayscale histogram of our pixel intensities of our
    # image is bi-modal, which simply means that the histogram is two peaks.

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # apply Otsu's automatic thresholding -- Otsu's method automatically
    # determines the best threshold value `T` for us
    (T, thresh_inv) = cv2.threshold(blurred, 0, 255,
                                    cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    cv2.imshow("Threshold", thresh_inv)
    print("Otsu's thresholding value: {}".format(T))

    cv2.imshow("Output", cv2.bitwise_and(image, image, mask=thresh_inv))


def adaptive_threshold(image):
    # In adaptive thresholding, sometimes called local thresholding,
    # our goal is to statistically examine the pixel intensity values in the neighborhood of a given pixel p.
    # However, choosing the size of the pixel neighborhood for local thresholding is absolutely crucial.

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray,(7, 7), 0)

    # instead of manually specifying the threshold value, we can use adaptive
    # thresholding to examine neighborhoods of pixels and adaptively threshold
    # each neighborbood -- in this example, we'll calculate the mean value
    # of the neighborhood area of 11 pixels and threshold based on that value;
    # finally, our constant C is subtracted from the mean calculation (in this
    # case 4)
    opencv_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 15)
    cv2.imshow("Opencv Thresh", opencv_thresh)

    # personally, I prefer the scikit-image adaptive thresholding, it just
    # feels a lot more "Pythonic"
    skimage_thresh = threshold_adaptive(blurred, 30, offset=5).astype(np.uint8) * 255
    skimage_thresh = cv2.bitwise_not(skimage_thresh)
    cv2.imshow("Scikit image Mean Thresh", skimage_thresh)



if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to image")
    args = vars(ap.parse_args())

    img = cv2.imread(args["image"])
    cv2.imshow("Original", img)

    simple_thresholding(img, 10)
    # otsu_threshold(img)
    # adaptive_threshold(img)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()

