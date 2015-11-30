# -- coding: utf-8 --

__author__ = 'amryfitra'


import numpy as np
import cv2


canvas = np.zeros((300, 300, 3), dtype=np.uint8)

green = (0, 255, 0)

cv2.line(canvas, (0, 0), (300, 300), green, 2)

cv2.rectangle(canvas, (10, 10), (60, 60), green, -1)

cv2.imshow("Drawing", canvas)

canvas_circle = np.zeros((300, 300, 3), dtype=np.uint8)

(w, h) = canvas_circle.shape[1], canvas_circle.shape[0]
(cX, cY) = int(w/2), int(h/2)
cv2.circle(canvas_circle, (cX, cY), 20, (0, 0, 255), 5)

cv2.imshow("Drawing Circle", canvas_circle)

cv2.waitKey(0)