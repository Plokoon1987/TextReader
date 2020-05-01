import cv2
import numpy as np
import pytesseract


def getting_better_text(img, threshold):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, gray = cv2.threshold(gray, threshold, 255, cv2.THRESH_TOZERO)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    th = cv2.dilate(th, None, iterations=1)
    th = cv2.erode(th, None, iterations=1)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV)
    return th

def getting_red_box(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_color = np.array([0,50,50]) # Red Ini
    upper_color = np.array([10,255,255]) # Red Ini
    mask = cv2.inRange(hsv, lower_color, upper_color)

    lower_color = np.array([170,50,50]) # Red End
    upper_color = np.array([180,255,255]) # Red End
    mask += cv2.inRange(hsv, lower_color, upper_color)

    masked = cv2.bitwise_and(img, img, mask=mask)

    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)

#    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    gray = cv2.GaussianBlur(gray, (1, 1), 0)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=3)
    return thresh

img = cv2.imread('img.png')
better_text = getting_better_text(img, 200)
red_box = getting_red_box(img)

contours, hierarchy = cv2.findContours(red_box, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
print(len(contours))
cv2.drawContours(better_text, contours, -1, (0, 255, 0), 3)

#cv2.imshow('Image', img)
cv2.imshow('Better Text', better_text)
#cv2.imshow('Red Box', red_box)

cv2.waitKey(0)
cv2.destroyAllWindows()

#img = cv2.imread('breakingnews.png')
#print(pytesseract.image_to_string(img))
print(pytesseract.image_to_string(better_text))
