import cv2
import numpy as np
import pytesseract

class SuperImage:
    color_ranges = {
        'red': ([[0, 80, 80], [10, 255, 255]],
                [[170, 80, 80], [180, 255, 255]]
                )
    }

    def __init__(self, image, color):
        self.image = cv2.imread(image)
        self.sharp_image = self.unsharp_mask()
        self.color = color

    def unsharp_mask(self, repeat=1):
        img = self.image.copy()
        for ind in range(0, repeat):
            img_blur = cv2.GaussianBlur(img, (9, 9), 10)
            img = cv2.addWeighted(img, 1.5, img_blur, -0.5, 0, img)
        return img

    def get_mask(self):
        blur = cv2.medianBlur(self.sharp_image, 5)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

        mask = 0
        for rang in self.color_ranges[self.color]:
            COLOR_MIN = np.array([rang[0]], np.uint8)
            COLOR_MAX = np.array([rang[1]], np.uint8)
            mask += cv2.inRange(hsv, COLOR_MIN, COLOR_MAX)
        return mask

    def get_rectangle(self):
        _, th = cv2.threshold(self.get_mask(), 200, 255, 0)
        contours, hierarchy = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        return cv2.minAreaRect(contours[max_index])

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

        dst = cv2.warpPerspective(self.sharp_image, M, size)
        return dst

    def refining(self):
        pers = self.pers_transf()
        gray = cv2.cvtColor(pers, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        _, th = cv2.threshold(gray, 194, 255, cv2.THRESH_BINARY)

        th = cv2.dilate(th, None, iterations=1)
        th = cv2.erode(th, None, iterations=1)

        _, th = cv2.threshold(th, 1, 255, cv2.THRESH_BINARY_INV)
        return th



img = SuperImage('img.png', 'red')
refined = img.refining()

cv2.imshow('IMG', img.image)
cv2.imshow('Refined', refined)

cv2.waitKey(0)
cv2.destroyAllWindows()

print(pytesseract.image_to_string(refined))
