#!/usr/local/bin/python3
# coding: UTF-8
# Author: David
# Email: youchen.du@gmail.com
# Created: 2017-04-14 17:47
# Last modified: 2017-04-14 18:32
# Filename: test_open_cv.py
# Description:
import cv2
import numpy as np

fname = 'LP_data/dataset/P1/G9/R3_l.png'

img = cv2.imread(fname, 0)
ret, dst = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
_, contours, hierarchy = cv2.findContours(dst, 1, 2)
dst = cv2.cvtColor(dst, cv2.COLOR_GRAY2RGB)

cnt = contours[0]
M = cv2.moments(cnt)

cx = int(M['m10'] / M['m00'])
cy = int(M['m01'] / M['m00'])

print(cx, cy)
# img[cy][cx] = 0

area = cv2.contourArea(cnt)

for c in contours:
    perimeter = cv2.arcLength(c, True)
    epsilon = 0.005 * perimeter
    approx = cv2.approxPolyDP(c, epsilon, True)
    cv2.drawContours(dst, [approx], -1, (0, 0, 255), 2)

hull = cv2.convexHull(cnt, returnPoints=False)
#defects = cv2.convexityDefects(cnt, hull)
#for i in range(defects.shape[0]):
#    s, e, f, d = defects[i, 0]
#    start = tuple(cnt[s][0])
#    end = tuple(cnt[e][0])
#    far = tuple(cnt[f][0])
#    cv2.line(dst, start, end, (0, 0, 255), 2)

k = cv2.isContourConvex(cnt)

# Straight Bounding Rectangle
x, y, w, h = cv2.boundingRect(cnt)
# cv2.rectangle(dst, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Rotated Rectangle
rect = cv2.minAreaRect(cnt)
box = cv2.boxPoints(rect)
box = np.int0(box)
# cv2.drawContours(dst, [box], 0, (0, 0, 255), 2)

# Minimum Enclosing Circle
(x, y), radius = cv2.minEnclosingCircle(cnt)
center = (int(x), int(y))
radius = int(radius)
# cv2.circle(dst, center, radius, (0, 255, 0), 2)

# Fitting an Ellipse
ellipse = cv2.fitEllipse(cnt)
# cv2.ellipse(dst, ellipse, (0, 255, 0), 2)

# Fitting a Line
rows, cols = dst.shape[:2]
[vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0, 0.01, 0.01)
lefty = int((-x*vy/vx) + y)
righty = int(((cols-x)*vy/vx) + y)
# cv2.line(dst, (cols-1, righty), (0, lefty), (0, 255, 0), 2)

cv2.imshow('test', dst)
cv2.waitKey(0)
