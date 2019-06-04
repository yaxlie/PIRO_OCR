import cv2
import numpy as np
from PIL import Image, ImageEnhance

def show_image(image, name, x=0, y=0, wait=False):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, image)
    cv2.resizeWindow(name, 600, 600)
    cv2.moveWindow(name, x, y)
    if wait:
        cv2.waitKey(0)


class PreProcessing:
    def __init__(self, image, gamma=0.8, contrast=10, threshold=1, debug=False):
        self.original_image = np.copy(image)
        self.debug = debug
        self.gamma = gamma
        self.contrast = contrast
        self.threshold = threshold

        self.result_image = self.preprocess_image(image)

    def preprocess_image(self, image):
        g = self.gamma
        c = self.contrast
        t = self.threshold

        img = self.adjust_gamma(image, g)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(img)
        contrast = ImageEnhance.Contrast(pil_im)
        contrast = contrast.enhance(c)

        img = np.array(contrast)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = cv2.GaussianBlur(img, (5, 5), 1)

        image = self.apply_threshold(img, t)

        kernel = np.ones((5, 5), np.uint8)

        image = cv2.erode(image, kernel, iterations=1)
        # image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)



        if self.debug:
            show_image(img, 'Grey', 0, 300)

            show_image(image, 'Thresh', 800, 300)

        return image

    def adjust_gamma(self, image, gamma=1.0):
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")

        # apply gamma correction using the lookup table
        return cv2.LUT(image, table)

    def change_contrast(self, img, level):
        factor = (259 * (level + 255)) / (255 * (259 - level))

        def contrast(c):
            value = 128 + factor * (c - 128)
            return max(0, min(255, value))

        return img.point(contrast)

    def apply_threshold(self, image, t):
        thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, t)

        return thresh


def pre_process(image):
    return image
