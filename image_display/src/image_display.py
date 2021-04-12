#!/usr/bin/env python
import cv2
import numpy as np

image = cv2.imread('interaction.jpeg')
image2 = cv2.imread('wave.png')

# I just resized the image to a quarter of its original size
image = cv2.resize(image, (0, 0), None, .5, .5)
image2 = cv2.resize(image2, (0, 0), None, .5, .5)


numpy_horizontal_concat = np.concatenate((image, image2), axis=1)

cv2.imshow('image', numpy_horizontal_concat)
cv2.moveWindow('image', 2000,0) 

cv2.waitKey()