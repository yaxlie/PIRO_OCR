import cv2
import numpy as np
import keras

import matplotlib.pyplot as plt

from PIL import Image, ImageEnhance

from skimage.transform import resize
from skimage.color import rgb2gray

DEBUG = False


def show_image(image, name, x=0, y=0, wait=False):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, image)
    cv2.resizeWindow(name, 600, 600)
    cv2.moveWindow(name, x, y)
    if wait:
        cv2.waitKey(0)


def crop_img(img, line):
    reshaped_line = np.reshape(line, (-1, 2))
    topx, topy = np.min(reshaped_line, axis=0)
    bottomx, bottomy = np.max(reshaped_line, axis=0)
    return img[topy:bottomy + 1, topx:bottomx + 1]


def sample_img(img, step_size=1):
    start_idx = 0
    end_idx = img.shape[1]
    window_size = img.shape[0]

    samples = []

    while start_idx + window_size <= end_idx:

        sample = img[:window_size, start_idx:start_idx + window_size]
        sample = rgb2gray(resize(sample, (28, 28, 3), anti_aliasing=True))
        sample = sample.reshape((*sample.shape, 1))

        samples.append(sample)

        start_idx += step_size

    return np.array(samples)



class RecognizeNumbers:
    model_filename = 'rec_digits.h5'

    def __init__(self):
        self.model = keras.models.load_model('saved_models/' + self.model_filename)
        self.currently_processed = None

    def predict(self, sampled_imgs):
        probs = self.model.predict(sampled_imgs, steps=1)
        max_probs = np.max(probs, axis=1)

        # Dropp if no numbers
        if np.median(max_probs) < 1e-1:
            return None

        return np.argmax(probs, axis=1)

    def numbers_hist(self, img, lines):
        for line in lines:
            for elem in line:
                if len(elem):
                    cropped_img = crop_img(img, elem)
                    sampled_imgs = sample_img(cropped_img)
                    if sampled_imgs is not None and len(sampled_imgs) > 0:
                        classes = self.predict(sampled_imgs)

                        plt.imshow(cropped_img)
                        plt.show()
                        pass

class LinesUtil:
    def __init__(self, image, pp_image):
        self.image = image.copy()
        self.pp_image = pp_image
        self.contours, hierarchy = cv2.findContours(pp_image, 1, 2)

        self.hull = []
        for cnt in self.contours:
            # epsilon = 0.1 * cv2.arcLength(contour, True)
            # approx = cv2.approxPolyDP(contour, epsilon, True)
            self.hull.append(cv2.convexHull(cnt, False))

    def get_lines(self):
        MAX_DIFF = 80
        MIN_DIFF = 20
        MAX_LINE_DIFF = 40
        words = []
        lines = []
        last = 10000
        for h in self.hull:
            min_y = min(point[0][1] for point in h)
            max_y = max(point[0][1] for point in h)
            diff_y = max_y - min_y
            avg_y = (min_y + max_y) / 2
            if MIN_DIFF < diff_y < MAX_DIFF:
                if last - avg_y > MAX_LINE_DIFF:
                    lines.append(words)
                    words = []
                    last = avg_y
                words.append(h)
                if DEBUG:
                    print(avg_y)
        i = 0
        if DEBUG:
            self.crop_lines(lines)
            for line in lines:
                cv2.drawContours(self.image, line, -1, (255 * (i % 2), 120 * (i % 3), 50 * (i % 6)), 3)

        return lines

    def crop_lines(self, lines):
        i = 0

        for line in lines:
            i += 1

            # try to crop contour to the new image
            # for word in lines:
            mask = np.zeros_like(self.pp_image)  # Create mask where white is what we want, black otherwise
            cv2.drawContours(mask, line, -1, 255, 3)
            out = np.copy(self.image)  # Extract out the object and place into output image
            out[mask == 255] = self.image[mask == 255]

            mask = np.zeros_like(self.pp_image)  # Create mask where white is what we want, black otherwise
            cv2.drawContours(mask, line, -1, 255, 3)
            out = np.copy(self.image)  # Extract out the object and place into output image
            out[mask == 255] = self.image[mask == 255]

            # Now crop
            (y, x) = np.where(mask == 255)
            if len(y) > 0 and len(x) > 0:
                (topy, topx) = (np.min(y), np.min(x))
                (bottomy, bottomx) = (np.max(y), np.max(x))
                out = out[topy:bottomy + 1, topx:bottomx + 1]

                # Show the output image
                if DEBUG:
                    cv2.imshow('Output', out)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()


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

        img = image
        # img = self.adjust_gamma(image, g)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # pil_im = Image.fromarray(img)
        # contrast = ImageEnhance.Contrast(pil_im)
        # contrast = contrast.enhance(c)
        #
        # img = np.array(contrast)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = cv2.GaussianBlur(img, (5, 5), 1)

        image = self.apply_threshold(img, t)

        kernel = np.ones((5, 5), np.uint8)

        image = 255 - image

        image = cv2.dilate(image, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12)))
        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1)))
        image = cv2.dilate(image, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1)))
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 6)))

        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15)))
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 6)))

        # image = cv2.erode(image, cv2.getStructuringElement(cv2.MORPH_RECT, (8, 1)), iterations=2)
        # image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

        image = 255 - image

        if self.debug:
            # show_image(img, 'Grey', 0, 300)

            show_image(image, 'Thresh', 800, 300, True)

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
        thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, t)

        return thresh


def pre_process(image):
    return image
