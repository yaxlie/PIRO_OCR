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
    pp_image = processing.PreProcessing(image, params['gamma'], params['contrast'], params['threshold']).result_image
    linesUtil = processing.LinesUtil(image, pp_image)

    lines = linesUtil.get_lines()

    processing.show_image(pp_image, 'Image', 0, 0, False)
    processing.show_image(linesUtil.image, 'Image2', 700, 0, DEBUG)
