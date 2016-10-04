import cv2
import numpy as np

im = cv2.imread('distillation1stfloor1.jpg')
imres = cv2.resize(im,None,fx=.25,fy=.25,interpolation = cv2.INTER_LINEAR)

imgray = cv2.cvtColor(imres,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,127,255,0)

image, contours, heirarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

cnt = contours[-1]

#epsilon = 0.1*cv2.arcLength(cnt,True)
#approx = cv2.approxPolyDP(cnt,epsilon,True)


img = cv2.drawContours(imres, [cnt], 0, (0,255,0), 3)

cv2.imshow('Polygons', img)
