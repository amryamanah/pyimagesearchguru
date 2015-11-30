
import numpy as np
import cv2
from IPython import embed

def tic_tac_toe():
    # load the tic-tac-toe image and convert it to grayscale
    image = cv2.imread("../img/tictactoe.png")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find all contours on the tic-tac-toe board
    (_, cnts, _) = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop over the contour
    cv2.drawContours(image, cnts, -1, (0, 255, 0))
    for (i, c) in enumerate(cnts):
        # compute the area of the contour along with the bounding box to compute aspec ration
        area = cv2.contourArea(c)
        (x, y, w, h) = cv2.boundingRect(c)

        # compute the convex hull of the contour, then use the area of the
        # original contour and the area of the convex hull to compute the
        # solidity
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        solidity = area / float(hull_area)

        # initialize the character text
        char = "?"

        # if solidity high then "O"
        if solidity > 0.9:
            char = "O"

        # otherwise if solidity it still reasonable high, then "X"
        elif solidity > 0.5:
            char = "X"

        if char != "?":
            cv2.drawContours(image, [c], -1, (255, 255, 0), 3)
            cv2.putText(image, char, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 255, 0), 4)
        print("{} (Contour #{}) -- solidity={:.2f}".format(char, i + 1, solidity))

    cv2.imshow("image", image)


def tetris():
    # load the tetris image, convert it to grayscale, and threshold the image
    image = cv2.imread("../img/tetris_blocks.png")
    # image = cv2.imread("../img/more_shapes_example.png")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)[1]
    # (T, thresh) = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    # Show the original and thresholded image
    cv2.imshow("Thresh", thresh)

    # find external contours in threshold image and allocate memory for the convex hull image
    (_, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull_img = np.zeros(gray.shape[:2], dtype=np.uint8)
    for (i, c) in enumerate(cnts):
        # compute the area of the contour along with bounding box area to compute the aspect ratio
        area = cv2.contourArea(c)
        (x, y, w, h) = cv2.boundingRect(c)

        # compute the aspect ratio of the contour, which is simply the width
        # divided by the height of the bounding box
        aspect_ratio = w / float(h)

        # compute the extent of the contour by dividing area of the contour by area of bounding box
        extent = area / float(w * h)

        # compute convex hull of the contour then use the area of the original contour and the area of the convex hull
        # to compute solidity
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        solidity = area / float(hull_area)

        # visualize the original contours and the convex hull and intialize the name of the shape
        cv2.drawContours(hull_img, [hull], -1, 255, -1)
        cv2.drawContours(image, [c], -1, (240, 0, 159), 3)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 3)

        shape = ""

        # if the aspect ratio is approximately one, then the shape is a square
        if 0.98 <= aspect_ratio <= 1.02:
            shape = "SQUARE"

        # if the width is 3x longer than the height, then we have a rectangle
        elif aspect_ratio >= 3.0:
            shape = "RECTANGLE"

        # if the extent is sufficiently small, then we have a L-piece
        elif extent < 0.65:
            shape = "L-PIECE"

        # if the solidity is sufficiently large enough, then we have a Z-piece
        elif solidity > 0.80:
            shape = "Z-PIECE"

        # visualize result
        cv2.putText(image, shape, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 0, 159), 2)
        print("Contour #{} - {}: aspect_ratio = {:.2f}, extent = {:.2f}, solidity = {:2f}".format(
            i + 1, shape, aspect_ratio, extent, solidity
        ))

        cv2.imshow("Image", image)
        cv2.imshow("Hull Image", hull_img)
        cv2.waitKey(0)


if __name__ == "__main__":
    #tic_tac_toe()
    tetris()

    cv2.waitKey(0)