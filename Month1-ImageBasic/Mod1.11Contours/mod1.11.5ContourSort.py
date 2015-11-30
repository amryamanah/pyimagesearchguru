import cv2
import numpy as np
import argparse
from IPython import embed


def sort_contours(cnts, method="left-to-rignt"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # Handle if we need to sort in revers
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to
    # bottom
    bounding_boxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, bounding_boxes) = zip(*sorted(zip(cnts, bounding_boxes), key=lambda b: b[1][i], reverse=reverse))

    return cnts, bounding_boxes


def draw_contour(image, c, i):
    # compute the center of the contour area and draw a circle
    # representing the center
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    # draw contour number
    cv2.putText(image, "#{}".format(i + 1), (cX -20, cY), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.drawContours(image, [c], -1, (255, 255, 0), 2)
    return image

if __name__ == '__main__':
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True)
    ap.add_argument("-m", "--method", required=True, help="Sorting method")
    args = vars(ap.parse_args())

    # load the image and initialize the accumulated edge image
    print(args["image"])
    image = cv2.imread(args["image"])
    accum_edged = np.zeros(image.shape[:2], dtype=np.uint8)

    # loop over the blue, green, and red channels, respectively
    for chan in cv2.split(image):
        # blur the channel, extract edges from it, and accumulate the set
        # of edges for the image
        chan = cv2.medianBlur(chan, 11)
        edged = cv2.Canny(chan, 50, 200)
        # cv2.imshow("Edged", edged)
        # cv2.imshow("Accum Edge", accum_edged)
        accum_edged = cv2.bitwise_or(accum_edged, edged)
        # cv2.waitKey(0)

    # show the accumulated edge map
    cv2.imshow("Edge Map", accum_edged)

    # find contours in the accumulated image, keeping only the largest ones
    (_, cnts, _) = cv2.findContours(accum_edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    orig = image.copy()

    # loop over the (unsorted) contours and draw them
    for (i, c) in enumerate(cnts):
        orig = draw_contour(orig, c, i)

    # show the original, unsorted contour image
    cv2.imshow("Unsorted", orig)

    # sort the contours according to the provided method
    (cnts, bounding_boxes) = sort_contours(cnts, method=args["method"])

    # loop over the (now sorted) contours and draw them
    for (i, c) in enumerate(cnts):
        draw_contour(image, c, i)

    # show the output image
    cv2.imshow("Sorted", image)
    cv2.waitKey(0)



