import cv2
import numpy as np
#from matplotlib import pyplot as plt
import pytesseract

img = cv2.imread('img.png')

# Creating Gray image
gray_imgs = {}
gray_imgs['ORIGINAL'] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, gray_imgs['GRAY_THRESH'] = cv2.threshold(gray_imgs['ORIGINAL'], 190, 255, cv2.THRESH_BINARY_INV)
text = pytesseract.image_to_string(gray_imgs['GRAY_THRESH'])
print(text)
'''
labels = ['ORIGINAL', 'GRAY_THRESH']
for i, label in enumerate(labels):
    plt.subplot(2,2,i+1),
    plt.title(labels[i])
    plt.imshow(gray_imgs[label])
    plt.xticks([])
    plt.yticks([])

plt.show()


# Creating HSV image

imgs = {}
imgs['ORIGINAL'] = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_color = np.array([0,50,50]) # Red Ini
upper_color = np.array([10,255,255]) # Red Ini
mask = cv2.inRange(imgs['ORIGINAL'], lower_color, upper_color)

lower_color = np.array([170,50,50]) # Red End
upper_color = np.array([180,255,255]) # Red End
mask += cv2.inRange(imgs['ORIGINAL'], lower_color, upper_color)

imgs['MASK'] = mask

imgs['RESULT'] = cv2.bitwise_and(img, img, mask=imgs['MASK'])

_, imgs['THRESH'] = cv2.threshold(imgs['RESULT'], 220, 255, cv2.THRESH_BINARY)

imgs['GRAY'] = cv2.cvtColor(imgs['THRESH'], cv2.COLOR_BGR2GRAY)


contours, _ = cv2.findContours(imgs['GRAY'], cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
print("Number of contours = " + str(len(contours)))

cv2.drawContours(imgs['ORIGINAL'], contours, -1, (0, 255, 0), 3)

labels = ['ORIGINAL', 'MASK', 'RESULT', 'THRESH', 'GRAY']
for i, label in enumerate(labels):
    plt.subplot(2,3,i+1), 
    plt.title(labels[i])
    plt.imshow(imgs[label])
    plt.xticks([])
    plt.yticks([])


plt.show()

cv2.drawContours(gray_imgs['GRAY_THRESH'], contours, -1, (0, 255, 0), 3)
'''

cv2.imshow('Image Gray', gray_imgs['GRAY_THRESH'])

cv2.waitKey(0)
cv2.destroyAllWindows()
