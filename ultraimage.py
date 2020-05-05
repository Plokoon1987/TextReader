import cv2
import numpy as np
import pytesseract

class DocumentOcr(object):
    def extract_code(self, image):
        # Sharpening Image
        img = image.copy()
        img_blur = cv2.GaussianBlur(img, (15, 15), 10)
        sharp_image = cv2.addWeighted(img, 1.5, img_blur, -0.5, 0, img)

        gray = cv2.cvtColor(sharp_image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        thresh = cv2.bitwise_and(img, img, mask=thresh)
#        gray = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
#        ret, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_TOZERO_INV)

#        thresh = cv2.GaussianBlur(thresh, (15, 15), 0)
#        thresh = cv2.addWeighted(img, 1.5, img_blur, -0.5, 0, img)

        thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(thresh, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        thresh = cv2.bitwise_and(img, img, mask=thresh)

        gray = cv2.cvtColor(thresh, cv2.COLOR_BGR2HSV)
#        gray = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
#        blur = cv2.GaussianBlur(gray, (5, 5), 10)
        ret, th = cv2.threshold(thresh, 75, 255, cv2.THRESH_TOZERO)
#        thresh = cv2.bitwise_and(th, th, mask=thresh)
#        ret, th = cv2.threshold(thresh, 100, 255, cv2.THRESH_TOZERO_INV)
#        thresh = cv2.bitwise_and(img, img, mask=thresh)

#        ret, thresh = cv2.threshold(thresh, 100, 255, cv2.THRESH_TOZERO_INV)




#        ret, thresh = cv2.threshold(thresh, 100, 255, cv2.THRESH_BINARY_INV)

#        for x in range(2):
#            thresh = cv2.erode(thresh, None, iterations=3)
#            thresh = cv2.dilate(thresh, None, iterations=3)
#            thresh = cv2.dilate(thresh, None, iterations=1)
#            thresh = cv2.erode(thresh, None, iterations=1)

        gray = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, contours, -1, (0, 255, 0), 3)


#        thresh = cv2.Canny(thresh, 50, 150, apertureSize = 3)
#        lines = cv2.HoughLines(thresh, 1, np.pi/90, 200)
#        print(lines)

#        for rho,theta in lines[0]:
#            a = np.cos(theta)
#            b = np.sin(theta)
#            x0 = a*rho
#            y0 = b*rho
#            x1 = int(x0 + 1000*(-b))
#            y1 = int(y0 + 1000*(a))
#            x2 = int(x0 - 1000*(-b))
#            y2 = int(y0 - 1000*(a))
#
#            cv2.line(thresh, (x1,y1), (x2,y2), (0,0,255), 2)

#        for x in range(1):
#            thresh = cv2.erode(thresh, None, iterations=1)
#            thresh = cv2.dilate(thresh, None, iterations=1)

#        thresh = cv2.GaussianBlur(thresh, (5, 5), 0)
#        imgray = cv2.GaussianBlur(imgray, (5, 5), 0)
#        ret, thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#        thresh = cv2.adaptiveThreshold(imgray, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)


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
