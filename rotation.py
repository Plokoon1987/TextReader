import cv2
import numpy as np

def get_blur(img):
    return cv2.medianBlur(img, 5)

def get_hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

def get_mask(img, color_ranges):
    mask = 0
    for rang in color_ranges:
        COLOR_MIN = np.array([rang[0]], np.uint8)
        COLOR_MAX = np.array([rang[1]], np.uint8)
        mask += cv2.inRange(img, COLOR_MIN, COLOR_MAX)
    return mask

def get_thresh(img):
    _, th = cv2.threshold(img, 127, 255, 0)
    return th

def get_contours(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def get_max_contour(contours):
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    return contours[max_index]

def get_rectangle(contour):
    x,y,w,h = cv2.boundingRect(contour)
    return [(x, y), (x+w, y+h)]

img = cv2.imread('img.png')
blur = get_blur(img)
hsv = get_hsv(blur)
red1 = [[0, 80, 80], [10, 255, 255]]
red2 = [[170, 80, 80], [180, 255, 255]]
mask = get_mask(hsv, [red1, red2])
thresh = get_thresh(mask)
contours = get_contours(thresh)
max_cont = get_max_contour(contours)
rectangle = cv2.minAreaRect(max_cont)
center, size, angle = rectangle
if angle < -45:
    angle = 90 + angle
else:
    angle = -angle

rows, cols, _ = img.shape
M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
dst = cv2.warpAffine(img, M, (cols, rows))

box = cv2.boxPoints(rectangle)
box = np.int0(box)
for point in box:
    cv2.circle(img, tuple(point), 8, (0, 255, 0), -1)

cv2.imshow('Output', dst)

cv2.waitKey(0)
cv2.destroyAllWindows()

