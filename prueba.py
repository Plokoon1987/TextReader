import cv2
import numpy as np
import pytesseract


def getting_better_text(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, th = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY_INV) # Trying to get pure white colors 
#    ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#    gray = cv2.GaussianBlur(gray, (5, 5), 0)
#    ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return th


'''
blur = cv2.medianBlur(img, 5)
#ret, th = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(blur,
                            255,
                            cv2.ADAPTIVE_THRESH_MEAN_C,
                            cv2.THRESH_BINARY_INV,
                            11,
                            2)
th3 = cv2.adaptiveThreshold(blur,
                            255,
                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                            cv2.THRESH_BINARY_INV,
                            11,
                            2)

cv2.imshow('Image', th2)

cv2.waitKey(0)
cv2.destroyAllWindows()
'''
img = cv2.imread('img.png')
better_text = getting_better_text(img)

cv2.imshow('Image', img)
cv2.imshow('Better Text', better_text)

cv2.waitKey(0)
cv2.destroyAllWindows()

img = cv2.imread('breakingnews.png')
print(pytesseract.image_to_string(img))
