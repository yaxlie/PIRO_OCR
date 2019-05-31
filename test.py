import io
import sys

import cv2
import yaml
from matplotlib.image import imread

import Sliders
import processing

DEBUG = True

sliders = Sliders.Sliders()

image = imread("{}".format(sys.argv[1]))

while True:
    pp_image = processing.PreProcessing(image, sliders.gamma, sliders.contrast, sliders.mean, DEBUG).result_image
    key = cv2.waitKey(0)

    # press 's' button to save params to .yaml file
    if key == 115:
        data = {
            'gamma': sliders.gamma,
            'contrast': sliders.contrast,
            'threshold': sliders.mean,
                }

        # Write YAML file
        with io.open('params.yaml', 'w', encoding='utf8') as outfile:
            yaml.dump(data, outfile, default_flow_style=False, allow_unicode=True)


