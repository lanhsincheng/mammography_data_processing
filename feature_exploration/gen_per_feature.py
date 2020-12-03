"""
    generate different features of original images and save images
    Args:
        path (str):
"""
read_image_path = r'D:\Mammograph\ROI_training_dataset\preprocess/'
write_image_path = r'D:\Mammograph\gabor_training_dataset\JPEGImages_genfromseg/'
frequency = 0.2 #0.1
theta = 0

import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from scipy import ndimage as ndi

from skimage import data
from skimage.util import img_as_float
from skimage.filters import gabor_kernel


def compute_feats(image, kernels):
    feats = np.zeros((len(kernels), 2), dtype=np.double)
    for k, kernel in enumerate(kernels):
        filtered = ndi.convolve(image, kernel, mode='wrap')
        feats[k, 0] = filtered.mean()
        feats[k, 1] = filtered.var()
    return feats

def power(image, kernel):
    # Normalize images for better comparison.
    image = (image - image.mean()) / image.std()
    return np.sqrt(ndi.convolve(image, np.real(kernel), mode='wrap')**2 +
                   ndi.convolve(image, np.imag(kernel), mode='wrap')**2)
count = 0
images = os.listdir(read_image_path)
for image in images:
    # prepare filter bank kernels
    frequency = frequency
    theta = theta
    theta = theta / 4. * np.pi
    # kernel = np.real(gabor_kernel(frequency, theta=theta))
    kernel = gabor_kernel(frequency, theta=theta)
    rd_path = read_image_path + image
    # shrink = (slice(0, None, 3), slice(0, None, 3))
    img = img_as_float(cv2.imread(rd_path, 0))

    # img = cv2.imread(rd_path, 0) # add blur
    # img = cv2.medianBlur(img, 3)

    # prepare reference features
    gabor_img = power(img, kernel)
    gabor_img = gabor_img/(gabor_img.max())
    # gabor_img = cv2.cvtColor(gabor_img, cv2.COLOR_GRAY2BGR)
    # cv2.imshow('gabor', gabor_img)
    # cv2.waitKey(0)
    wb_path = write_image_path + image
    gabor_img = gabor_img*255
    count += 1
    print(count)
    cv2.imwrite(wb_path, cv2.merge((gabor_img, gabor_img, gabor_img)))
