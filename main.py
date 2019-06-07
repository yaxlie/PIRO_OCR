import io

import numpy as np
import cv2
import os
import yaml

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

    MAX_DIFF = 80
    MIN_DIFF = 20
    MAX_LINE_DIFF = 15
    words = []
    lines = []
    last = 10000
    for h in hull:
        min_y = min(point[0][1] for point in h)
        max_y = max(point[0][1] for point in h)
        diff_y = max_y - min_y
        avg_y = (min_y + max_y) / 2
        if MIN_DIFF < diff_y < MAX_DIFF:
            if last - avg_y > MAX_LINE_DIFF:
                lines.append(words)
                words = []
            words.append(h)
            last = avg_y

            print(avg_y)
    i = 0
    for line in lines:
        i += 1
        cv2.drawContours(img, line, -1, (255 * (i % 2), 120 * (i % 3), 50 * (i % 6)), 3)

    processing.show_image(pp_image, 'Image', 0, 0, False)
    processing.show_image(img, 'Image2', 700, 0, DEBUG)
