import numpy as np
import argparse
import uuid
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
                help="Path to the output directory")
ap.add_argument("-n", "--num-images", type=int, default=500,
                help="# num of distractor images to generate")
args = vars(ap.parse_args())

os.makedirs(args["output"])
# loop over the number of distractor images to generate
for i in range(0, args["num_images"]):
    # allocate memory for the image, then generate the (x, y)
    # center of the circle, then generate the radius of the
    # circle, ensuring that the circle is fully contained in
    # the image
    image = np.zeros((500, 500, 3), dtype="uint8")
    center = np.random.uniform(low=105, high=405, size=(2,)).astype("int0")
    r = np.random.uniform(low=25, high=100, size=(1,)).astype("int0")[0]

    color = np.random.uniform(low=0, high=255, size=(3,))
    print(color)
    cv2.circle(image, tuple(center), r, color, -1)
    cv2.imwrite("{}/{}.jpg".format(args["output"], uuid.uuid4()), image)

image = np.zeros((500, 500, 3), dtype="uint8")
topLeft = np.random.uniform(low=25, high=225, size=(2,)).astype("int0")
botRight = np.random.uniform(low=250, high=400, size=(2,)).astype("int0")

color = np.random.uniform(low=0, high=255, size=(3,))
cv2.rectangle(image, tuple(topLeft), tuple(botRight), color, -1)
cv2.imwrite("{}/{}.jpg".format(args["output"], uuid.uuid4()), image)
