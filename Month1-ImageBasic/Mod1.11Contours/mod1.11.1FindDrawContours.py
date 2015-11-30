import cv2
import argparse
import numpy as np


def draw_all_contour(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    (_, cnts, _) = cv2.findContours(gray.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    clone = image.copy()
    cv2.drawContours(clone, cnts, -1, (0, 255, 0), 2)
    print("Found {} contours".format(len(cnts)))

    cv2.imshow("All contours", clone)

def iterate_contour(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    (_, cnts, _) = cv2.findContours(gray.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    clone = image.copy()
    print("Found {} contours".format(len(cnts)))

    for (i, c) in enumerate(cnts):
        print("Drawing contour #{}".format(i + 1))
        cv2.drawContours(clone, [c], -1, (0, 255, 0), 2)
        cv2.imshow("Single Contour", clone)
        cv2.waitKey(0)


def draw_only_external_contour(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clone = image.copy()
    (_, cnts, _) = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("Found {} External Contours".format(len(cnts)))

    cv2.drawContours(clone, cnts, -1, (0, 255, 0), 2)
    cv2.imshow("All Contours", clone)


def draw_contour_mask(image):
    clone = image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    (_, cnts, _) = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for (i, c) in enumerate(cnts):
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [c], -1, 255, -1)

        cv2.imshow("Image", clone)
        cv2.imshow("Mask", mask)
        cv2.imshow("Image + mask", cv2.bitwise_and(clone, clone, mask=mask))
        cv2.waitKey(0)


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
                    help='Path to image')
    args = vars(ap.parse_args())

    img = cv2.imread(args["image"])
    cv2.imshow("Original", img)

    # draw_all_contour(img)
    # iterate_contour(img)
    # draw_only_external_contour(img)
    draw_contour_mask(img)
    cv2.waitKey(0)
