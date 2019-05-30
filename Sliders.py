import cv2

class Sliders:
    def __init__(self):
        g = 0.8
        c = 10
        m = 1
        self.gamma = g
        self.contrast = c
        self.mean = m

        def onGammaChange(x):
            p = cv2.getTrackbarPos('gamma', 'sliders') +1
            self.gamma = g * (p/100)

            pass
        def onContrastChange(x):
            p = cv2.getTrackbarPos('contrast', 'sliders') +1
            self.contrast = c * (p / 100)
            pass
        def onMeanChange(x):
            p = cv2.getTrackbarPos('mean', 'sliders')
            self.mean = p
            pass

        cv2.namedWindow('sliders', cv2.WINDOW_AUTOSIZE)

        cv2.createTrackbar('gamma', 'sliders', 0, 10000, onGammaChange)
        cv2.createTrackbar('contrast', 'sliders', 0, 2000, onContrastChange)
        cv2.createTrackbar('mean', 'sliders', 0, 255, onMeanChange)

        cv2.setTrackbarPos('gamma', 'sliders', 140)
        cv2.setTrackbarPos('contrast', 'sliders', 100)
        cv2.setTrackbarPos('mean', 'sliders', 200)

        cv2.moveWindow("sliders", 1000, 0)

        pass