#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 16:52:25 2021

@author: houtan ghaffari
"""

import cv2
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from scipy import interpolate


# read the image
base_dir = '../input_images'
print("You need to first put the image you want to use in the input_images folder.")
filename = input("Please type the image's name(for example flower.jpg): ")
print(f"\nReading the file named {filename} ...")
path_to_image = os.path.join(base_dir, filename)
while not os.path.isfile(path_to_image):
    filename = input("\nThis file does not exist. Please enter a valid name or q to exit: ")
    if filename=='q':
        sys.exit()
    path_to_image = os.path.join(base_dir, filename)
print(f"\nRead {filename} successfully!")
img = cv2.imread(path_to_image)
    
# I'm going to work with images of at most 640 width or height
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
    return img

print("\nResizing the image into ", end=' ')
img = resize_img(img)

coordinates = []
# mouse callback function
def get_point(event,x,y,flags,param):
    if event == cv2.EVENT_FLAG_LBUTTON:
        print(f'({x},{y})', end=' ')
        cv2.circle(img,(x,y),2,(255,0,0),-1)
        coordinates.append([x, y])
        
print("\nPlease select the points of interest using left click on the image window.")
print("Press esc to quit when you were done.\n")
print("\nSelected Points:")
cv2.namedWindow('image')
cv2.setMouseCallback('image', get_point)
while(1):
    cv2.imshow('image', img)
    if cv2.waitKey(5) & 0xFF == 27:
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        break
prefix = filename.split('.')[0]
base_dir_to_save = os.path.join('../outputs/', prefix)
if not os.path.exists(base_dir_to_save):
    os.mkdir(base_dir_to_save)
path_to_save = os.path.join(base_dir_to_save, prefix + '_coordinates.npy')
print(f"\nSaving the coordinates in {path_to_save}")
np.save(path_to_save, coordinates)
print('Saving completed.')

print('Drawing the Spline...')
coordinates = np.load(path_to_save)
y = coordinates[:,1]
x = coordinates[:,0]
sr = 3 # scale of the increase in resolution

# X:
t = np.linspace(0, 1, len(x)) # t is a dummy variable to help the interpolation
tck_x = interpolate.splrep(t, x, s=0)
np.save(os.path.join(base_dir_to_save, 'tck_x_' + prefix + '.npy'), tck_x)
t = np.linspace(0, 1, sr * len(tck_x[1]))
x_interpolated = interpolate.splev(t, tck_x, der=0)

# Y:
t = np.linspace(0, 1, len(y))
tck_y = interpolate.splrep(t, y, s=0)
np.save(os.path.join(base_dir_to_save, 'tck_y_' + prefix + '.npy'), tck_y)
t = np.linspace(0, 1, sr * len(tck_y[1]))
y_interpolated = interpolate.splev(t, tck_y, der=0)

# polynomial which I want to draw base on the interpolated coordinates
x_vertices = x_interpolated.astype('int32')
y_vertices = y_interpolated.astype('int32')
pts = np.concatenate((x_vertices[:, None], y_vertices[:, None]), axis=1)
pts = np.expand_dims(pts, 1)

# draw
temp = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
# color = [np.random.randint(0, 255) for _ in range(3)]
temp = cv2.polylines(temp, [pts], False, (255,0,127), 4)
plt.imshow(temp)
plt.axis('off')
plt.show()
temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
cv2.imwrite(os.path.join(base_dir_to_save, 'spline_' + prefix + '.jpg'), temp)
print("Finished!")

