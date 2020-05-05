import cv2
import numpy as np
import pytesseract

class DocumentOcr(object):
    def extract_code(self, image):
        # Sharpening Image
        img = image.copy()
        blurred = cv2.GaussianBlur(img, (9, 9), 10)
        sharp_image = cv2.addWeighted(img, 1.5, blurred, -0.5, 0, img)
        sharp_image = img

        gray = cv2.cvtColor(sharp_image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 90, 255, cv2.THRESH_TOZERO)
        new_image = cv2.bitwise_and(img, img, mask=thresh)

        gray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 110, 255, cv2.THRESH_TOZERO_INV)
        new_image = cv2.bitwise_and(img, img, mask=thresh)

        gray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 95, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=1)
        thresh = cv2.erode(thresh, None, iterations=2)

        new_image = cv2.bitwise_and(img, img, mask=thresh)

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        rect = cv2.minAreaRect(contours[max_index])

        cv2.drawContours(new_image, contours, max_index, (0, 255, 0), 1)

        old_box = cv2.boxPoints(rect)
        old_box = [[round(val[0]), round(val[1])] for val in old_box]
        old_box = np.float32(old_box)

        center, new_box, angle = rect
        new_box = [round(val) for val in reversed(new_box)]
        size = tuple(new_box)
        new_box = [[0, 0], [new_box[0], 0], [0, new_box[1]], [new_box[0], new_box[1]]]
        new_box = [[250+elem[0], 250+elem[1]] for elem in new_box]
        new_box = sorted(new_box, key=lambda x: (-x[1], -x[0]))
        last_elem = new_box.pop(-2)
        new_box = new_box + [last_elem]
        new_box = np.float32(new_box)


        M = cv2.getPerspectiveTransform(old_box, new_box)


        dst = cv2.warpPerspective(img, M, (1000,1000))


        return dst


        return thresh

#        # Masking Red
#        img_blur = cv2.medianBlur(sharp_image, 5)
#        hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)
#
#        COLOR_MIN = np.array([0, 80, 80], np.uint8)
#        COLOR_MAX = np.array([10, 255, 255], np.uint8)
#        mask = cv2.inRange(hsv, COLOR_MIN, COLOR_MAX)
#
#        COLOR_MIN = np.array([170, 80, 80], np.uint8)
#        COLOR_MAX = np.array([180, 255, 255], np.uint8)
#        mask += cv2.inRange(hsv, COLOR_MIN, COLOR_MAX)
#
#        # Getting Rectangle points
#        _, th = cv2.threshold(mask, 200, 255, 0)
#        contours, hierarchy = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
#        areas = [cv2.contourArea(c) for c in contours]
#        max_index = np.argmax(areas)
#        rect = cv2.minAreaRect(contours[max_index])
#
#        # Rotate and Trim Image
#        old_box = cv2.boxPoints(rect)
#        old_box = [[round(val[0]), round(val[1])] for val in old_box]
#        old_box = np.float32(old_box)
#
#        center, new_box, angle = rect
#        new_box = [round(val) for val in reversed(new_box)]
#        size = tuple(new_box)
#        new_box = [[0, 0], [new_box[0], 0], [0, new_box[1]], new_box ]
#        new_box = sorted(new_box, key=lambda x: (-x[1], -x[0]))
#        last_elem = new_box.pop(-2)
#        new_box = new_box + [last_elem]
#        new_box = np.float32(new_box)
#
#        M = cv2.getPerspectiveTransform(old_box, new_box)
#
#        dst = cv2.warpPerspective(sharp_image, M, size)
#
#        # Refining Image
#        gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
#        gray = cv2.GaussianBlur(gray, (3, 3), 0)
#
#        _, th = cv2.threshold(gray, 194, 255, cv2.THRESH_BINARY)
#
#        th = cv2.dilate(th, None, iterations=1)
#        th = cv2.erode(th, None, iterations=1)
#
#        _, th = cv2.threshold(th, 1, 255, cv2.THRESH_BINARY_INV)
#        return th

#image = cv2.imread('img.png')
#a = DocumentOcr().extract_code(image)
#
#cv2.imshow('IMG', a)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#print(pytesseract.image_to_string(refined))

images = [['sample0.png', 'U62QMKI'],
          ['sample1.png', 'SGSO9MU'],
          ['sample2.png', '8HBBNEY6'],
          ['sample3.png', 'MOROYKHN'],
          ]

for image in images:
    img = cv2.imread(image[0])
    a = DocumentOcr().extract_code(img)
    cv2.imshow(image[0], a)

cv2.waitKey(0)
cv2.destroyAllWindows()
