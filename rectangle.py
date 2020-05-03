import cv2
import numpy as np

def blur(img):
    return cv2.medianBlur(img, 5)

def hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

def mask(img, color_ranges):
    mask = 0
    for rang in color_ranges:
        COLOR_MIN = np.array([rang[0]], np.uint8)
        COLOR_MAX = np.array([rang[1]], np.uint8)
        mask += cv2.inRange(img, COLOR_MIN, COLOR_MAX)
    return mask

def thresh(img):
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

img_orig = cv2.imread('img.png')
img = blur(img_orig)
img = hsv(img)
red1 = [[0, 80, 80], [10, 255, 255]]
red2 = [[170, 80, 80], [180, 255, 255]]
img = mask(img, [red1, red2])
img = thresh(img)
contours = get_contours(img)
max_cont = get_max_contour(contours)
rectangle = get_rectangle(max_cont)

cv2.rectangle(img_orig, rectangle[0], rectangle[1], (0, 255, 0), 2)
cv2.imshow('Output', img_orig)

cv2.waitKey(0)
cv2.destroyAllWindows()

