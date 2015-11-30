# -- coding: utf-8 --
import cv2


class LabHistogram(object):
    def __init__(self, bins):
        # store the number of bins for the histogram
        self.bins = bins

    def describe(self, image, mask=None):
        # convert the image to the L*a*b* color space, compute a histogram, and normalize it
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        hist = cv2.calcHist([lab], [0, 1, 2], mask, self.bins, [0, 256, 0, 256, 0, 256])
        # normalization ensure the width and height of our input image has no effect on the output histogram
        hist = cv2.normalize(hist, hist).flatten()
        return hist
