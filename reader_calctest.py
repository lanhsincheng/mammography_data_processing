import os
from os import walk
import glob
import cv2

rd_path =  r'G:\DDSM\CBIS-DDSM\JPEG_test\Calc/'
dirs = os.listdir(rd_path)
for file in dirs:
    next_level = rd_path + file + '/'
    nexts = os.listdir(next_level)
    for folder_img in nexts:
        nextnext = next_level  + folder_img + '/'
        final_img = glob.glob(nextnext + "*.jpg")
        original_name = final_img[0]
        img = cv2.imread(original_name)
        dest = 'G:\DDSM\CBIS-DDSM\JPEG_test\Calc_trans/' + file + '.jpg'
        cv2.imwrite(dest, img)
        
