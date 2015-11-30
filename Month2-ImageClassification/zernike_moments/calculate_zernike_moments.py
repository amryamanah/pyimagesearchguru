import argparse
import mahotas
import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True)

args = vars(ap.parse_args())

image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)[1]

(_, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cv2.imshow("Thres", thresh)
print(len(cnts))
cnt = sorted(cnts, key=cv2.contourArea)[0]
(x, y, w, h) = cv2.boundingRect(cnt)
roi = gray[y:y+h, x:x+w]
zernike_mom = mahotas.features.zernike_moments(roi, 200, degree=3)
print(zernike_mom)

cv2.imshow("Thres", roi)
# cv2.drawContours(image, [cnt], -1, (255, 0, 0), -2)
box = cv2.minAreaRect(cnt)
box = np.int32(cv2.boxPoints(box))
cv2.drawContours(image, [box], -1, (0, 0, 255), 2)

cv2.imshow("Image", image)
cv2.waitKey(0)