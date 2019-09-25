
from skimage import img_as_float, img_as_ubyte
from skimage.feature import canny
from skimage.morphology import dilation, disk, rectangle, erosion
from skimage.transform import hough_line, hough_line_peaks, rotate
import numpy as np


def words_finding(image):
    image = img_as_float(image)

    # if image.shape[1] > image.shape[0]:
    #     image = rotate(image, angle=-90, resize=True)

    edges = canny(image)
    h, theta, d = hough_line(edges)
    peaks = hough_line_peaks(h, theta, d)

    angles = np.multiply(np.copy(peaks[1]), np.mod(np.copy(peaks[1]) + np.pi / 2, np.pi))
    angle = np.median(angles)
    image = rotate(image, angle=angle, resize=True)
    if image.shape[1] > image.shape[0]:
        image = rotate(image, angle=-90, resize=True)
    image = 1-image

    image = dilation(image, disk(4))
    image = erosion(image, rectangle(1, 16))
    image = dilation(image, disk(6))
    image = erosion(image, rectangle(24, 1))
    image = dilation(image, disk(2))
    # image = erosion(image, rectangle(1, 16))
    # image = dilation(image, disk(4))
    # image = erosion(image, rectangle(16, 1))
    # image = erosion(image, rectangle(8, 1))
    # image = dilation(image, disk(2))

    #plt.imshow(image, cmap=plt.cm.gray)
    #plt.show()
    return img_as_ubyte(1 - image)
