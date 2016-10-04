import numpy as np
import cv2

im = cv2.imread('distillation1stfloor1.jpg')
imres = cv2.resize(im,None,fx=.25,fy=.25,interpolation = cv2.INTER_LINEAR)
imgray = cv2.cvtColor(imres,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,127,255,0)
edge = cv2.Canny(imres,100,200)

image, contours1, heirarchy = cv2.findContours(edge,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
image, contours, heirarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

img = cv2.drawContours(imres, contours, -1, (0,255,0), 3)
img2 = cv2.drawContours(imres, contours1, -1, (0,255,0), 3)
cv2.imshow('threshold',img)
cv2.imshow('edge',img)



