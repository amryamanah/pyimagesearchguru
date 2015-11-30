from skimage import feature
from skimage import exposure
import cv2


img = cv2.imread("1.bmp")
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hue, sat, val = cv2.split(hsv_img)
b, g, r = cv2.split(img)

# blurred = cv2.GaussianBlur(sat, (15, 15), 0)
blurred = cv2.bilateralFilter(sat, 9, 75, 75)
edge = cv2.Canny(blurred, 10, 100)
cv2.imshow("Blurred", blurred)
# cv2.imshow("Edge", edge)

(H, hog_image) = feature.hog(blurred, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3),
                             normalise=True, visualise=True)
hog_image = exposure.rescale_intensity(hog_image, out_range=(0, 255))
hog_image = hog_image.astype("uint8")
cv2.imshow("HOG image", hog_image)
cv2.waitKey(0)
