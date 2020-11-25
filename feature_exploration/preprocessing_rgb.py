"""
        Image enhancement with rgb image
        Usage : set the img_path and choose the transform you want to do
        Args: img_path, write_img_path

        Returns: enhanced image
"""
import numpy as np
import cv2

# img_path = r'D:\Mammograph\ROI_training_dataset\JPEGImages\\Case_326_CC.jpg'
img_path = './dog.jpg'
img = cv2.imread(img_path)

# create a CLAHE object for RGB (Arguments are optional).
lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
lab_planes = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(6,6))
lab_planes[0] = clahe.apply(lab_planes[0])
lab = cv2.merge(lab_planes)
cl1 = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
median = cv2.medianBlur(cl1, 3)

# create median feature.
median3_im = cv2.medianBlur(img, 3)

# create power-law transform feature.
powerlaw_im = np.array(255*(cl1/255)**3.5, dtype='uint8')

# create log transform feature.
c = 255 / np.log(1 + np.max(powerlaw_im))
log_image = c * (np.log(powerlaw_im + 1))
log_im = np.array(log_image, dtype=np.uint8)

compare = np.concatenate((cl1, log_im), axis=1) #side by side comparison
cv2.imshow('img', compare)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('enhance2_dog.jpg',log_im)