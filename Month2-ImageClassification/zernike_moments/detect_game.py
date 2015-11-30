from scipy.spatial import distance as dist
import numpy as np
import mahotas
import cv2

def describe_shapes(image):
    # Initialize the list of shape feature
    shape_features = []

    # convert tihe image to grayscale, blur it, and threshold it
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (13, 13), 0)
    thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)[1]

    # perform a series of dilation and erosions to close holes in the shapes
    thresh = cv2.dilate(thresh, None, iterations=4)
    thresh = cv2.erode(thresh, None, iterations=2)

    (_, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # loop over the contour
    for c in cnts:
        mask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)

        (x, y, w, h) = cv2.boundingRect(c)
        roi = mask[y:y+h, x:x+w]

        features = mahotas.features.zernike_moments(roi, max(w/2, h/2), degree=8)
        shape_features.append(features)

    return cnts, shape_features

refImage = cv2.imread("pokemon_red.png")
(_, gameFeatures) = describe_shapes(refImage)

shapesImage = cv2.imread("shapes.png")
(cnts, shapeFeatures) = describe_shapes(shapesImage)

D = dist.cdist(gameFeatures, shapeFeatures)
print(D)
i = np.argmin(D)
for j, c in enumerate(cnts):
    if i != j:
        box = cv2.minAreaRect(c)
        box = np.int32(cv2.boxPoints(box))
        cv2.drawContours(shapesImage, [box], -1, (0, 0, 255), 2)

box = cv2.minAreaRect(cnts[i])
box = np.int32(cv2.boxPoints(box))
cv2.drawContours(shapesImage, [box], -1, (0, 255, 0), 2)
(x, y, w, h) = cv2.boundingRect(cnts[i])
cv2.putText(shapesImage, "Found", (x, y - 10), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.9, (0, 255, 0), 3)

cv2.imshow("Input Image", refImage)
cv2.imshow("Detected shapes", shapesImage)
cv2.waitKey(0)
