"""
    generate different features of original images and show in comparison
    Usage : gen_per_feature
    Args:
        path (str):
"""

read_image_path1 = r'D:\Mammograph\YOLO_ROI_training_dataset\feature\original_malignant/0_Case_14_CC.jpg'
read_image_path2 = r'D:\Mammograph\YOLO_ROI_training_dataset\feature\original_malignant/0_Case_70_CC.jpg'
read_image_path3 = r'D:\Mammograph\YOLO_ROI_training_dataset\feature\original_malignant/0_Case_289_CC.jpg'
read_image_path4 = r'D:\Mammograph\YOLO_ROI_training_dataset\feature\original_benign/1_Case_27_CC.jpg'
read_image_path5 = r'D:\Mammograph\YOLO_ROI_training_dataset\feature\original_benign/1_Case_74_CC.jpg'
read_image_path6 = r'D:\Mammograph\YOLO_ROI_training_dataset\feature\original_benign/1_Case_103_MLO.jpg'

write_image_path = r'D:\Mammograph\YOLO_ROI_training_dataset\feature\train/'

import matplotlib.pyplot as plt
import numpy as np
import cv2
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


def match(feats, ref_feats):
    min_error = np.inf
    min_i = None
    for i in range(ref_feats.shape[0]):
        error = np.sum((feats - ref_feats[i, :])**2)
        if error < min_error:
            min_error = error
            min_i = i
    return min_i


# prepare filter bank kernels
kernels = [] # len = 16
for theta in range(4):
    theta = theta / 4. * np.pi
    for sigma in (1, 3):
        for frequency in (0.05, 0.25):
            kernel = np.real(gabor_kernel(frequency, theta=theta,
                                          sigma_x=sigma, sigma_y=sigma))
            kernels.append(kernel)


shrink = (slice(0, None, 3), slice(0, None, 3))
mal1 = img_as_float(cv2.imread(read_image_path1, 0))[shrink]
mal2 = img_as_float(cv2.imread(read_image_path2, 0))[shrink]
mal3 = img_as_float(cv2.imread(read_image_path3, 0))[shrink]
ben1 = img_as_float(cv2.imread(read_image_path4, 0))[shrink]
ben2 = img_as_float(cv2.imread(read_image_path5, 0))[shrink]
ben3 = img_as_float(cv2.imread(read_image_path6, 0))[shrink]
image_names = ('mal1', 'mal2', 'mal3', 'ben1', 'ben2', 'ben3')
images = (mal1, mal2, mal3, ben1, ben2, ben3)

# prepare reference features
ref_feats = np.zeros((6, len(kernels), 2), dtype=np.double)
ref_feats[0, :, :] = compute_feats(mal1, kernels)
ref_feats[1, :, :] = compute_feats(mal2, kernels)
ref_feats[2, :, :] = compute_feats(mal3, kernels)
ref_feats[3, :, :] = compute_feats(ben1, kernels)
ref_feats[4, :, :] = compute_feats(ben2, kernels)
ref_feats[5, :, :] = compute_feats(ben3, kernels)

print('Rotated images matched against references using Gabor filter banks:')

print('original: mal1, rotated: 30deg, match result: ', end='')
feats = compute_feats(ndi.rotate(mal1, angle=190, reshape=False), kernels)
print(image_names[match(feats, ref_feats)])

print('original: mal2, rotated: 70deg, match result: ', end='')
feats = compute_feats(ndi.rotate(mal2, angle=70, reshape=False), kernels)
print(image_names[match(feats, ref_feats)])

print('original: mal3, rotated: 70deg, match result: ', end='')
feats = compute_feats(ndi.rotate(mal3, angle=70, reshape=False), kernels)
print(image_names[match(feats, ref_feats)])

print('original: ben1, rotated: 145deg, match result: ', end='')
feats = compute_feats(ndi.rotate(ben1, angle=145, reshape=False), kernels)
print(image_names[match(feats, ref_feats)])

print('original: ben2, rotated: 145deg, match result: ', end='')
feats = compute_feats(ndi.rotate(ben2, angle=145, reshape=False), kernels)
print(image_names[match(feats, ref_feats)])

print('original: ben3, rotated: 145deg, match result: ', end='')
feats = compute_feats(ndi.rotate(ben3, angle=145, reshape=False), kernels)
print(image_names[match(feats, ref_feats)])



def power(image, kernel):
    # Normalize images for better comparison.
    image = (image - image.mean()) / image.std()
    return np.sqrt(ndi.convolve(image, np.real(kernel), mode='wrap')**2 +
                   ndi.convolve(image, np.imag(kernel), mode='wrap')**2)

# Plot a selection of the filter bank kernels and their responses.
results = []
kernel_params = []
for theta in (0, 1, 2):
    theta = theta / 4. * np.pi
    for frequency in (0.1, 0.4, 1.6):
        kernel = gabor_kernel(frequency, theta=theta)
        params = 'theta=%d,\nfrequency=%.2f' % (theta * 180 / np.pi, frequency)
        kernel_params.append(params)
        # Save kernel and the power image for each image
        results.append((kernel, [power(img, kernel) for img in images]))

fig, axes = plt.subplots(nrows=10, ncols=7, figsize=(10, 7)) #5 6
plt.gray()

fig.suptitle('Image responses for Gabor filter kernels', fontsize=12)

axes[0][0].axis('off')

# Plot original images
for label, img, ax in zip(image_names, images, axes[0][1:]):
    ax.imshow(img)
    ax.set_title(label, fontsize=9)
    ax.axis('off')

for label, (kernel, powers), ax_row in zip(kernel_params, results, axes[1:]):
    # Plot Gabor kernel
    ax = ax_row[0]
    ax.imshow(np.real(kernel))
    ax.set_ylabel(label, fontsize=5)
    ax.set_xticks([])
    ax.set_yticks([])

    # Plot Gabor responses with the contrast normalized for each filter
    # vmin = np.min(powers)
    # vmax = np.max(powers)
    for patch, ax in zip(powers, ax_row[1:]):
        ax.imshow(patch, vmin=None, vmax=None)
        ax.axis('off')

plt.show()