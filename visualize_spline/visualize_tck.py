#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 21:45:03 2021

@author: houtan
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import interpolate
import os
import sys

print("tck files are saved inside the outputs folder.")
print("You need to first put an image inside the input_images folder.")
print("Then, you have to run the spline.py file and extract the tck_x and tck_y.")
print("It's automatic, you just need to select your points on the image by mouse.")
print("If you have done this already, please enter the name of the image. (Try eagle.jpg or aquarium.jpg for a demo):")
filename = input()
prefix = filename.split('.')[0]
base_dir = '../outputs'
path_to_tck_x = os.path.join(base_dir, prefix, 'tck_x_' + prefix + '.npy')
path_to_tck_y = os.path.join(base_dir, prefix, 'tck_y_' + prefix + '.npy')
while not (os.path.isfile(path_to_tck_x) and os.path.isfile(path_to_tck_y)):
    filename = input("\nThis file does not exist. Please enter a valid name or q to exit: ")
    if filename=='q':
        sys.exit()
    prefix = filename.split('.')[0]
    path_to_tck_x = os.path.join(base_dir, prefix, 'tck_x_' + prefix + '.npy')
    path_to_tck_y = os.path.join(base_dir, prefix, 'tck_y_' + prefix + '.npy')

tck_x = np.load(path_to_tck_x, allow_pickle=True)
t, c, k = tck_x
tck_x = (t,c,k)
tck_y = np.load(path_to_tck_y, allow_pickle=True)
t, c, k = tck_y
tck_y = (t, c, k)

sr = 3
t = np.linspace(0, 1, sr * len(tck_x[1]))
x_interpolated = interpolate.splev(t, tck_x, der=0)
t = np.linspace(0, 1, sr * len(tck_y[1]))
y_interpolated = interpolate.splev(t, tck_y, der=0)
x_vertices = x_interpolated.astype('int32')
y_vertices = y_interpolated.astype('int32')
pts = np.concatenate((x_vertices[:, None], y_vertices[:, None]), axis=1)
pts = np.expand_dims(pts, 1)

image_dir = '../input_images'
path_to_image = os.path.join(image_dir, filename)
img = cv2.imread(path_to_image)
def resize_img(img):
    """
    This function resizes an image to width or height of 640px max, 
    and it respects the aspect ratio

    args:
        original_img - input image of shape (height, width, channels)

    returns:
        resized image
    """

    # if the size is already appropriate, pass
    if max(img.shape) <= 640:
        return img

    # take the size of heigh, width, and channels
    h, w, _ = img.shape

    if h > 640:
        r = 640 / h # to scale the other dimension properly
        dim = (int(r * w), 640)
    else:
        r = 640 / w
        dim = (640, int(r * h))
    print(dim)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)
    return img

print("\nResizing the image into ", end=' ')
img = resize_img(img)
color = [np.random.randint(0, 255) for _ in range(3)]
temp = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
temp = cv2.polylines(temp, [pts], False, color, 4)
plt.imshow(temp)
plt.axis('off')
plt.show()
