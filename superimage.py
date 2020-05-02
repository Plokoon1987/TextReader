import cv2
import numpy as np

class SuperImage:
    def __init__(self, image, color):
        self.image = cv2.imread(image)
        self.color = color

    def get_blur(self):
        return cv2.medianBlur(self.image, 5)

    def get_hsv(self):
        return cv2.cvtColor(self.get_blur(), cv2.COLOR_BGR2HSV)

    def get_mask(self):
        color_ranges = {
            'red': ([[0, 80, 80], [10, 255, 255]],
                    [[170, 80, 80], [180, 255, 255]]
                    )
        }
        mask = 0
        for rang in color_ranges[self.color]:
            COLOR_MIN = np.array([rang[0]], np.uint8)
            COLOR_MAX = np.array([rang[1]], np.uint8)
            mask += cv2.inRange(self.get_hsv(), COLOR_MIN, COLOR_MAX)
        return mask

    def get_thresh(self):
        _, th = cv2.threshold(self.get_mask(), 200, 255, 0)
        return th

    def get_contours(self):
        contours, hierarchy = cv2.findContours(self.get_thresh(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def get_max_contour(self):
        areas = [cv2.contourArea(c) for c in self.get_contours()]
        max_index = np.argmax(areas)
        return self.get_contours()[max_index]

    def get_rectangle(self):
        return cv2.minAreaRect(self.get_max_contour())

    def rotate(self):
        center, box, angle = self.get_rectangle()
        center = round(center[0]), round(center[1])
        rev_center = center[1], center[0]

        cv2.circle(self.image, center, 8, (0, 255, 0), -1)
        angle = 90 + angle if angle < -45 else -angle
        size = self.image.shape
        size = ( round(size[1]), round(size[0]) )

        M = cv2.getRotationMatrix2D(rev_center, angle, 1)
        return cv2.warpAffine(self.image, M, size)


img = SuperImage('img.png', 'red')
print(img.get_rectangle())
rot = img.rotate()

#cv2.imshow('Blur', rot)
cv2.imshow('Blur', img.image)
cv2.imshow('Rot', img.rotate())

cv2.waitKey(0)
cv2.destroyAllWindows()

