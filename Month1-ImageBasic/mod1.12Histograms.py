# -- coding: utf-8 --
import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np


def grayscale_hist(image, mask=None, normalized=False):
    if len(image.shape) > 2:
        # convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # calculate histogram
    hist = cv2.calcHist([gray], [0], mask, [255], [0, 256])

    plt.figure()
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    plt.xlim([0, 256])

    if normalized:
        hist = hist / hist.sum()
        plt.title("Grayscale histogram (Normalized)")
    else:
        plt.title("Grayscale histogram")

    plt.plot(hist)


def color_histogram(image, title="Flattened color hist", mask=None):
    # grabs image channels, initialize tuple colors and figure
    chans = cv2.split(image)
    colors = ("b", "g", "r")
    plt.figure()
    plt.title(title)
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")

    # loop over image channels
    for (chan, color) in zip(chans, colors):
        # calculate histogram for current channel and plot it
        hist = cv2.calcHist([chan], [0], mask, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])


def hist2d(image):
    chans = cv2.split(image)
    # Move to 2 D histogram
    # number of bins in the histogram from 256 to 32 so we can
    # better visualize the results
    fig = plt.figure()

    # plot 2D color histogram for green and blue
    ax = fig.add_subplot(131)
    hist_gb = cv2.calcHist([chans[1], chans[0]], [0, 1], None, [32, 32], [0, 256, 0, 256])
    p = ax.imshow(hist_gb, interpolation="nearest")
    ax.set_title("2D Color histogram for G and B")
    plt.colorbar(p)

    # plot 2D color histogram for green and red
    ax = fig.add_subplot(132)
    hist_gr = cv2.calcHist([chans[1], chans[2]], [0, 1], None, [32, 32], [0, 256, 0, 256])
    p = ax.imshow(hist_gr, interpolation="nearest")
    ax.set_title("2D Color histogram for G and R")
    plt.colorbar(p)

    # plot 2D color histogram for blue and red
    ax = fig.add_subplot(133)
    hist_br = cv2.calcHist([chans[0], chans[2]], [0, 1], None, [32, 32], [0, 256, 0, 256])
    p = ax.imshow(hist_br, interpolation="nearest")
    ax.set_title("2D Color histogram for B and R")
    plt.colorbar(p)



def hist3d(image):
    # our 2D histogram could only take into account 2 out of the 3
    # channels in the image so now let's build a 3D color histogram
    # (utilizing all channels) with 8 bins in each direction -- we
    # can't plot the 3D histogram, but the theory is exactly like
    # that of a 2D histogram, so we'll just show the shape of the
    # histogram
    hist = cv2.calcHist([image], [0, 1, 2],
        None, [9, 16, 8], [0, 256, 0, 256, 0, 256])
    print("3D histogram shape: %s, with %d values" % (hist.shape, hist.flatten().shape[0]))


def hist_equal(image, grayscale=False):
    if grayscale:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        eq = cv2.equalizeHist(gray)

    else:
        (b, g, r) = cv2.split(image)

        b = cv2.equalizeHist(b)
        g = cv2.equalizeHist(g)
        r = cv2.equalizeHist(r)

        eq = cv2.merge([b, g, r])

    return eq


def masked_hist(image):


    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.rectangle(mask, (60, 290), (210, 390), 255, -1)
    cv2.imshow("Mask", mask)
    masked = cv2.bitwise_and(image, image, mask=mask)
    cv2.imshow("Masked", masked)

    color_histogram(image)
    color_histogram(image, title="Masked histogram", mask=mask)


if __name__ == '__main__':
    # create command line argument
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
                    help="Path to image")
    args = vars(ap.parse_args())

    # load image to memory
    img = cv2.imread(args["image"])
    cv2.imshow("Origin", img)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("Grayscale", gray)

    # grayscale_hist(img)
    # color_histogram(img)
    # hist2d(img)
    hist3d(img)
    # eq = hist_equal(img, grayscale=True)
    # print(eq[272, 146])
    # cv2.imshow("Eq", eq)
    # grayscale_hist(eq)

    # masked_hist(img)

    plt.show()
    cv2.waitKey(0)
