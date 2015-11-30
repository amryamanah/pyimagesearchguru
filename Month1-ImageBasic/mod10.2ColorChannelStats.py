# -- coding: utf-8 --

from scipy.spatial import distance as dist
from imutils import paths
import numpy as np
import cv2
import os
from IPython import embed

# Grab the list of image paths and initialize to the index to store the image filename and feature vector
image_paths = sorted(list(paths.list_images(os.path.join("img", "dinos"))))
index = {}

# loop over the image paths
for im_path in image_paths:
    # load the image and extract the filename
    image = cv2.imread(im_path)
    filename = os.path.basename(im_path)

    # extract the mean and standard deviation from each channel of the BGR image,
    # then update the index with the feature vector
    (means, stds) = cv2.meanStdDev(image)
    features = np.concatenate([means, stds]).flatten()
    index[filename] = features

query = cv2.imread(image_paths[0])
cv2.imshow("Query (trex_01.png)", query)
keys = sorted(index.keys())

# loop over the filenames in the dictionary
for (i, k) in enumerate(keys):
    # if this is the first image, ignore it
    if k == "trex_01.png":
        continue

    # load the current image and compute the Euclidean distance between the
    # query image (i.e. the 1st image) and the current image
    image = cv2.imread(image_paths[i])
    d = dist.euclidean(index["trex_01.png"], index[k])

    # display the distance between the query image and the current image
    cv2.putText(image, "%.2f" % d, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    cv2.imshow(k, image)

cv2.waitKey(0)
