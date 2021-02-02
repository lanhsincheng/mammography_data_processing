"""
        Image enhancement with grayscale image
        Usage : set the img_path and choose the transform you want to do
        Args: img_path, write_img_path

        Returns: enhanced image
"""
import numpy as np
import cv2
import os
from matplotlib import pyplot as plt

rd_path = r'D:\Mammograph\ROI_training_dataset\maskoverlap_trans/'
# rd_path = r'D:\Mammograph\ROI_training_dataset\JPEGImages/'
wb_path = r'D:\Mammograph\ROI_training_dataset\temp/'
# check_path = r'D:\Mammograph\ROI_training_dataset\test2/'


def CLAHE(img):
    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(6, 6)) # 6
    cl = clahe.apply(img)

    # cv2.imshow("original", img)
    # cv2.imshow("tophat", cl)
    # cv2.waitKey()
    return cl

def median(img):
    # create median feature.
    median = cv2.medianBlur(img, 3)
    return median

def log_transform(img):
    # create log transform feature.
    c = 255 / np.log(1 + np.max(img))
    log_image = c * (np.log(img + 1))
    log_im = np.array(log_image, dtype=np.uint8)

    # cv2.imshow("log_im", log_im)
    # cv2.waitKey()
    return log_im

def power_law_transform(img):
    # create power-law transform feature.
    powerlaw_im = np.array(255 * (img / 255) ** 3.5, dtype='uint8')#3.5
    cv2.imshow("original", img)
    cv2.imshow("powerlaw_im", powerlaw_im)
    cv2.waitKey()
    return powerlaw_im

def power_law_transform_tophat(img):
    # create power-law transform feature.
    powerlaw_im = np.array(255 * (img / 255) ** 2.5, dtype='uint8')#3.5
    # cv2.imshow("original", img)
    # cv2.imshow("powerlaw_im", powerlaw_im)
    # cv2.waitKey()
    return powerlaw_im

def adjust_gamma(image, gamma=0.5):
    # build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def threshold(img):
    # ret,th = cv2.threshold(img,38,255,cv2.THRESH_BINARY)
    # img = power_law_transform_tophat(img)
    # img = CLAHE(img)
    ret, th = cv2.threshold(img, 22, 255, cv2.THRESH_BINARY) #IN35 #CBIS38
    cv2.imshow("original", img)
    cv2.imshow("threshold", th)
    cv2.imwrite('threshold.jpg', th)
    cv2.waitKey()
    return th

def tophat(img):
    # Getting the kernel to be used in Top-Hat
    filterSize = (8, 8) #11
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                       filterSize)#ELLIPSE

    # Reading the image named 'input.jpg'
    # input_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Applying the Top-Hat operation
    tophat_img = cv2.morphologyEx(img,
                                  cv2.MORPH_TOPHAT,
                                  kernel)

    cv2.imshow("original", img)
    cv2.imshow("tophat", tophat_img)
    cv2.imwrite('tophat_img.jpg', tophat_img)
    cv2.waitKey()
    return tophat_img

def erosion(img):
    kernel = np.ones((1, 1), np.uint8)
    erode_im = cv2.erode(img, kernel, iterations=1)

    cv2.imshow("original", img)
    cv2.imshow("erode_im", erode_im)
    cv2.imwrite('erode_im.jpg', erode_im)
    cv2.waitKey()
    return erode_im

def dilation(img):
    kernel = np.ones((1, 1), np.uint8) #3*3
    dilation_im = cv2.dilate(img, kernel, iterations=1)

    # cv2.imshow("original", img)
    # cv2.imshow("erode_im", dilation_im)
    # cv2.waitKey()
    return dilation_im

def hist_eqa(img):
    equ = cv2.equalizeHist(img)

    # cv2.imshow("original", img)
    # cv2.imshow("equ", equ)
    # cv2.waitKey()
    return equ

def canny(img):
    canny = cv2.Canny(img, 30, 150)

    # cv2.imshow('Input', img)
    # cv2.imshow('Result', canny)
    # cv2.waitKey()

    return canny

def fft(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))

    return magnitude_spectrum

def overlap(origin_img, img):
    dst = cv2.addWeighted(origin_img, 0.8, img, 0.2, 0)#CBIS-DDSM 0.8,0.2
    # cv2.imshow("original", origin_img)
    # cv2.imshow("overlap", dst)
    # cv2.waitKey()
    cv2.imwrite('dst.jpg', dst)
    return dst

def fusion(check, img2):
    rows = img2.shape[0]
    cols = img2.shape[1]

    check2 = check.copy()
    for i in range(rows):
        for j in range(cols):
            if img2[i, j] == 255:
                check2[i, j] = img2[i, j]
            else:
                check2[i, j] = check[i, j]

    # cv2.imshow("original", check)
    # cv2.imshow("pixeladd", check2)
    # cv2.waitKey()

    return check2

def pixeladd(origin_img, img):
    rows = origin_img.shape[0]
    cols = origin_img.shape[1]
    for i in range(rows):
        for j in range(cols):
            add_val = (int(origin_img[i,j])*0.2 + int(img[i,j])*0.8)/2
            if  add_val >= 255:
                img[i,j] = 255
            # elif 100 > add_val > 80:
            #     img[i, j] = add_val + 20
            else:
                img[i, j] = min(origin_img[i, j], img[i, j])+20

    # cv2.imshow("original", origin_img)
    # cv2.imshow("pixeladd", img)
    # cv2.waitKey()

    return img

images = os.listdir(rd_path)
for image in images:
    img_path = rd_path + image
    img = cv2.imread(img_path, 0)
    origin_img = img.copy()

    ## choose method  median->tophat->threshold
    img = median(img)
    # img = hist_eqa(img)
    # img = power_law_transform(img)
    img = adjust_gamma(img, gamma=0.4)
    p = img + origin_img
    # for i in range(origin_img.shape[0]):
    #     for j in range(origin_img.shape[1]):
    #         if p[i][j]>255 :
    #             p[i][j] = 255
    cv2.imshow("original", origin_img)
    cv2.imshow("adjust gamma", img)
    cv2.imwrite('adjust gamma.jpg', img)
    cv2.waitKey()


    # img2 = img.copy()
    # img = CLAHE(img)
    # img = pixeladd(origin_img, img)

    img = tophat(img)

    # img2 = canny(img2)

    # img = threshold(img)
    # img = fft(img)

    # img = log_transform(img)
    # img = power_law_transform(img)

    # check = overlap(img, img2)
    # img = tophat(img)

    img = threshold(img)
    # img = dilation(img)
    img = erosion(img)
    # check = img
    check = overlap(origin_img, img)
    # check2 = fusion(check, img2)
    # check = overlap(img2, img)

    wb_img_path = wb_path + image
    # check_img_path = check_path + image
    cv2.imwrite(wb_img_path,check)
    # cv2.imwrite(check_img_path, check)