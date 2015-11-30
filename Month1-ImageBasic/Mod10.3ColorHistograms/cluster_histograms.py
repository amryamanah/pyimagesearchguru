# -- coding: utf-8 --
from pyimagesearch.descriptors.labhistogram import LabHistogram
from sklearn.cluster import KMeans
from imutils import paths
import numpy as np
import argparse
import cv2


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True)
ap.add_argument("-k", "--clusters", type=int, default=2,
                help="# of clusters to generate")

args = vars(ap.parse_args())

# initialize the image descriptor along with the image matrix
desc = LabHistogram([8, 8, 8])
data = []

# grab image paths from dataset directory
image_paths = list(paths.list_images(args["dataset"]))
image_paths = np.array(sorted(image_paths))

# loop over the input dataset of images
for im_path in image_paths:
    # load the image, describe the image, then update the list of data
    image = cv2.imread(im_path)
    hist = desc.describe(image)
    data.append(hist)

# cluster the color histograms
clt = KMeans(n_clusters=args["clusters"])
labels = clt.fit_predict(data)


from IPython import embed
embed()

print(np.unique(labels))
# from IPython import embed
# embed()
for label in np.unique(labels):
    # grab all image paths that are assigned to the current label
    label_paths = image_paths[np.where(labels == label)]

    # loop over the image paths that belong to the current label
    for (i, path) in enumerate(label_paths):
        # load the image and display it
        image = cv2.imread(path)
        cv2.imshow("Cluster {}, Image #{}".format(label + 1, i + 1), image)

    cv2.waitKey(0)
