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
        center = [round(val) for val in center]

        angle = 90 + angle if angle < -45 else -angle
        size = self.image.shape
        size = ( round(size[1]), round(size[0]) )

        cv2.circle(img.image, tuple(center), 8, (0, 0, 255), -1) # TODO
        M = cv2.getRotationMatrix2D(tuple(center), angle, 1)
        dst = cv2.warpAffine(self.image, M, size)

        cv2.circle(dst, tuple(center), 5, (0, 255, 0), -1) # TODO
        return dst

    def pers_transf(self):
        rect = self.get_rectangle()
        old_box = cv2.boxPoints(rect)
        old_box = [[round(val[0]), round(val[1])] for val in old_box]
        old_box = np.float32(old_box)

        center, new_box, angle = rect
        new_box = [round(val) for val in reversed(new_box)]
        size = tuple(new_box)
        new_box = [[0, 0], [new_box[0], 0], [0, new_box[1]], new_box ]
        new_box = sorted(new_box, key=lambda x: (-x[1], -x[0]))
        last_elem = new_box.pop(-2)
        new_box = new_box + [last_elem]
        new_box = np.float32(new_box)

        M = cv2.getPerspectiveTransform(old_box, new_box)

        dst = cv2.warpPerspective(self.image, M, size)
        return dst



img = SuperImage('img.png', 'red')

pers = img.pers_transf()
gray = cv2.cvtColor(pers, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (3, 3), 0)

_, th = cv2.threshold(gray, 194, 255, cv2.THRESH_BINARY)

#_, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

th = cv2.dilate(th, None, iterations=1)
th = cv2.erode(th, None, iterations=1)

_, th = cv2.threshold(th, 1, 255, cv2.THRESH_BINARY_INV)



cv2.imshow('Pers', th)
cv2.imshow('Gray', gray)

cv2.waitKey(0)
cv2.destroyAllWindows()

import pytesseract
print(pytesseract.image_to_string(th))
