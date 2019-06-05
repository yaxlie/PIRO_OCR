import io

import numpy as np
import cv2
import os
import yaml
from scipy import ndimage

import processing

DEBUG = True

imgs = []

# Read YAML file
with open("params.yaml", 'r') as stream:
    params = yaml.safe_load(stream)

# Load in the images
for filepath in os.listdir('res/'):
    imgs.append(cv2.imread('res/{0}'.format(filepath), cv2.IMREAD_COLOR))

for image in imgs:
    if image.shape[1] > image.shape[0]:
        image = ndimage.rotate(image, angle=-90)

    img = image.copy()
    pp_image = processing.PreProcessing(image, params['gamma'], params['contrast'], params['threshold']).result_image
    contours, hierarchy = cv2.findContours(pp_image, 1, 2)

    hull = []
    for cnt in contours:
        # epsilon = 0.1 * cv2.arcLength(contour, True)
        # approx = cv2.approxPolyDP(contour, epsilon, True)
        hull.append(cv2.convexHull(cnt, False))

        # _, _, h, w = cv2.boundingRect(cnt)
        # epsilon = 0.1*cv2.arcLength(cnt,True)
        # vertices = cv2.approxPolyDP(cnt, epsilon, True)
        # vertices = cv2.convexHull(cnt, clockwise=True)
        # cv2.drawContours(img, [vertices], -1, (255, 255, 0), 3)

    hull2 = [s for s in hull if len(s) > 10]
    # print(hull2[0])
    cv2.drawContours(img, hull2, -1, (255, 255, 0), 3)

    processing.show_image(pp_image, 'Image', 0, 0, False)
    processing.show_image(img, 'Image2', 700, 0, DEBUG)
