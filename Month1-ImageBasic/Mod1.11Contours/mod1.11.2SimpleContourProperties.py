# -- coding: utf-8 --
import numpy as np
import cv2
import argparse


def draw_minimum_enclosing_circle(image):
    clone = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    (_, cnts, _) = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for i, c in enumerate(cnts):
        # compute area and perimeter of the contour
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        print("Contour #{} -- area: {:.2f}, perimeter: {:.2f}".format(
            (i + 1), area, perimeter
        ))

        ((x, y), radius) = cv2.minEnclosingCircle(c)
        cv2.circle(clone, (int(x), int(y)), int(radius), (0, 255, 0), 2)
        print(radius)
        # compute the center of the contour and draw contour number
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cv2.putText(clone, "#{}".format(i + 1), (cX - 20, cY), cv2.FONT_HERSHEY_COMPLEX, 1.25, (255, 255, 255), 4)

    cv2.imshow("Contours", clone)


def calculate_centroid(image):
    clone = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    (_, cnts, _) = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in cnts:
        # compute the moments of the contour which can be used to compute the
        # centroid or "center of mass" of the region
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        print("Center cx = {}, cy = {}".format(cX, cY))
        # draw the center of the contour on the image
        cv2.circle(clone, (cX, cY), 10, (0, 0, 255), -1)

        cv2.imshow("Centroids", clone)
        cv2.waitKey(0)


def calculate_area_perimeter(image):
    clone = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    (_, cnts, _) = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for i, c in enumerate(cnts):
        # compute area and perimeter of the contour
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        print("Contour #{} -- area: {:.2f}, perimeter: {:.2f}".format(
            (i + 1), area, perimeter
        ))

        cv2.drawContours(clone, [c], -1, (0, 255, 0), 2)

        # compute the center of the contour and draw contour number
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cv2.putText(clone, "#{}".format(i + 1), (cX - 20, cY), cv2.FONT_HERSHEY_COMPLEX, 1.25, (255, 255, 255), 4)

    cv2.imshow("Contours", clone)


def draw_bounding_box(image):
    clone = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    (_, cnts, _) = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        # fit a bounding box to the contour
        (x, y, w, h) = cv2.boundingRect(c)
        print(str(x) + "|" +str(y) + "|" +str(w) + "|" + str(h))
        cv2.rectangle(clone, (x,y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("Contours", clone)


def draw_rotated_bounding_box(image):
    clone = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    (_, cnts, _) = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        box = cv2.minAreaRect(c)
        rect = np.int32(cv2.boxPoints(box))
        cv2.drawContours(clone, [rect], -1, (0, 255, 0), 2)

    cv2.imshow("Rotated bounding box", clone)


def draw_ellipse(image):
    clone = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    (_, cnts, _) = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        if len(c) > 5:
            ellipse = cv2.fitEllipse(c)
            cv2.ellipse(clone, ellipse, (0, 255, 0), 2)

    cv2.imshow("Ellipse", clone)


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
                    help='Path to image')
    args = vars(ap.parse_args())

    img = cv2.imread(args["image"])
    cv2.imshow("Original", img)

    #calculate_centroid(img)
    #calculate_area_perimeter(img)
    #draw_bounding_box(img)
    # draw_rotated_bounding_box(img)
    draw_minimum_enclosing_circle(img)
    # draw_ellipse(img)
    cv2.waitKey(0)