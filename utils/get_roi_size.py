"""
    get each roi size and write these info into csv
    Usage :
    Args:
        path (str): file path which contain roi you want to get info
"""
import csv
import os
import numpy as np
import cv2

rd_path = r'G:\DDSM\CBIS-DDSM\JPEG_val/'

imgs = os.listdir(rd_path)

def write_csv(info_list):
    with open('size_info_Calc_trans_val.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for info in info_list:
            writer.writerow(info)

info_list = []
for img in imgs:
    roi_img = cv2.imread(rd_path + img)
    (height, weight, channel) = roi_img.shape
    area = height*weight
    ratio = weight/height
    info_list.append([img, height, weight, area, ratio])
write_csv(info_list)
