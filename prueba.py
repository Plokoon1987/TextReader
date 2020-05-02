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

def get_edges(img):
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grey = cv2.GaussianBlur(grey, (7, 7), 0)
    edged = cv2.Canny(grey, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)
    return edged

def getting_red_box(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_color = np.array([0,125,125]) # Red Ini
    upper_color = np.array([10,255,255]) # Red Ini
    mask = cv2.inRange(hsv, lower_color, upper_color)

    lower_color = np.array([170,125,125]) # Red End
    upper_color = np.array([180,255,255]) # Red End
    mask += cv2.inRange(hsv, lower_color, upper_color)

    masked = mask
    masked = cv2.erode(masked, None, iterations=1)
    masked = cv2.dilate(masked, None, iterations=1)
    return masked


def getting_outer(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours[0]

'''
def getting_extremes(item):
    def get_ext_point(arr, val, points, direct='x'):
        if direct == 'x':
            mask = arr[:,0] == (val, -1)
            filtered = arr[np.where(mask[:,0]==True)]
        else:
            mask = arr[:,0] == (-1, val)
            filtered = arr[np.where(mask[:,1]==True)]

        dists = []
        min_dist = 0
        ret = None
        for elem in filtered:
            p = tuple(elem[0])
            for point in points:
                dist = ( (p[0] - point[0])**2 + (p[1] - point[1])**2 )**0.5
                dists += [dist]

                print(dist)
                print(dists)
                import ipdb; ipdb.set_trace()
                if dist < min(dists):
                    min_dist = dist
                    ret = p

        return ret


    extl = tuple(item[item[:,:,0].argmin()][0])[0]
    extr = tuple(item[item[:,:,0].argmax()][0])[0]
    extt = tuple(item[item[:,:,1].argmin()][0])[1]
    extb = tuple(item[item[:,:,1].argmax()][0])[1]

    border_tl = (extl, extt)
    border_bl = (extl, extb)
    border_tr = (extr, extt)
    border_br = (extr, extb)

    extl_point = get_ext_point(item, extl, [border_tl, border_bl])
    extr_point = get_ext_point(item, extr, [border_tr, border_br])
#    extt_point = get_ext_point(item, extt, direct='y')
#    extb_point = get_ext_point(item, extb, direct='y')
    print(extl_point)
    print('')
    print(extr_point)
    print('')
#    print(extt_point)
#    print('')
#    print(extb_point)
#    print('')

#    for elem in extl_point:
#        print(tuple(elem))

    import ipdb; ipdb.set_trace()
#    return {'extl': extl, 'extr': extr, 'extt': extt, 'extb': extb}
    return {'border_tl': border_tl, 'border_bl': border_bl,
            'border_tr': border_tr, 'border_br': border_br}
'''


img = cv2.imread('img.png')
better_text = getting_better_text(img, 200)
red_box = getting_red_box(img)
outer = getting_outer(red_box)
cv2.drawContours(better_text, outer, -1, (0, 255, 0), 1)

'''
extremes = getting_extremes(outer)
for key, val in extremes.items():
    print(val)
    cv2.circle(better_text, val, 8, (0, 0, 255), -1)
extremes = getting_extremes(outer)
'''


#cv2.imshow('Image', img)
extremes = getting_extremes(outer)
cv2.imshow('Better Text', better_text)
cv2.imshow('Red Box', red_box)

cv2.waitKey(0)
cv2.destroyAllWindows()

#img = cv2.imread('breakingnews.png')
#print(pytesseract.image_to_string(img))
#print(pytesseract.image_to_string(better_text))
