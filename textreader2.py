import cv2
import numpy as np
from matplotlib import pyplot as plt

images = {}
images['ORIGINAL'] = cv2.imread('img.png')
images['GRAY'] = cv2.cvtColor(images['ORIGINAL'], cv2.COLOR_BGR2GRAY)
images['HSV'] = cv2.cvtColor(images['ORIGINAL'], cv2.COLOR_BGR2HSV)


lower_color = np.array([0,50,50]) # Red Ini
upper_color = np.array([10,255,255]) # Red Ini
mask = cv2.inRange(images['HSV'], lower_color, upper_color)

lower_color = np.array([170,50,50]) # Red End
upper_color = np.array([180,255,255]) # Red End
mask += cv2.inRange(images['HSV'], lower_color, upper_color)

images['MASK'] = mask

images['RESULT'] = cv2.bitwise_and(images['ORIGINAL'], images['ORIGINAL'], mask=images['MASK'])

_, images['THRESH'] = cv2.threshold(images['RESULT'], 200, 255, cv2.THRESH_BINARY)

#contours, _ = cv2.findContours(images['THRESH'], cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#
#print("Number of contours = " + str(len(contours)))

labels = ['ORIGINAL', 'GRAY', 'HSV', 'MASK', 'RESULT', 'THRESH']
for i, label in enumerate(labels):
    plt.subplot(2,3,i+1), 
    plt.title(labels[i])
    plt.imshow(images[label], 'gray')
    plt.xticks([])
    plt.yticks([])

plt.show()

#contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#print("Number of contours = " + str(len(contours)))
#
#cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
#cv2.imshow('New Img', img)
#
#cv2.waitKey(0)
#cv2.destroyAllWindows()
