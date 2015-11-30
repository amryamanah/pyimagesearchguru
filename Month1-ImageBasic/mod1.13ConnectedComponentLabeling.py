# -- coding: utf-8 --

from skimage.filters import threshold_adaptive
from skimage import measure
import numpy as np
import cv2
from IPython import embed

# load the license plate image from disk
plate = cv2.imread("img/license_plate.png")

# extract the Value component from the HSV color space and apply adaptive thresholding
# to reveal the characters on the license plate
V = cv2.split(cv2.cvtColor(plate, cv2.COLOR_BGR2HSV))[2]
cv2.imshow("Value channel", V)
thresh_first = threshold_adaptive(V, 30, offset=10)
thresh_second = thresh_first.astype("uint8") * 255
cv2.imshow("Thresh second", thresh_second)
thresh = cv2.bitwise_not(thresh_second)

# show the images
cv2.imshow("License Plate", plate)
cv2.imshow("Thresh", thresh)

labels = measure.label(thresh, neighbors=8, background=0)
mask = np.zeros(thresh.shape, dtype="uint8")

print("[INFO] found {} blobs".format(len(np.unique(labels))))

# Loop over unique components
for (i, label) in enumerate(np.unique(labels)):
    # if this is the background label, ignore it
    if label == -1:
        print("[INFO] label: -1 (background)")
        continue

    # otherwise, construct the label mask to display only connected components for
    # the current label
    print("[INFO] label: {} (foreground)".format(i))
    labelMask = np.zeros(thresh.shape, dtype="uint8")
    labelMask[labels == label] = 255
    numPixels = cv2.countNonZero(labelMask)
    # if the number of pixels in the component is sufficiently large, add it to our
    # mask of "large" blobs
    if numPixels > 300 and numPixels < 1500:
        mask = cv2.add(mask, labelMask)

    # show the label mask
    cv2.imshow("Label", labelMask)
    cv2.waitKey(0)

cv2.imshow("Large Blobs", mask)
cv2.waitKey(0)

cv2.destroyAllWindows()
