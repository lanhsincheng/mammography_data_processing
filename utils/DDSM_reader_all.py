## 解開所有層層環繞的資料夾
import os
from os import walk
import glob
import cv2

rd_path =  r'G:\DDSM\CBIS-DDSM\JPEG_test\Calc/'
dirs = os.listdir(rd_path)
count = 0
for file in dirs:
    if file.split('_')[-1] == 'CC' or file.split('_')[-1] == 'MLO': continue
    else:
        count += 1
        next_level = rd_path + file + '/'  # G:\DDSM\CBIS-DDSM\JPEG_test\Mass/Mass-Test_P_01510_RIGHT_MLO_1/
        nexts = os.listdir(next_level)
        for folder_img in nexts:
            nextnext = next_level  + folder_img + '/'
            nextnext_list = os.listdir(nextnext)
            for imgs in nextnext_list:
                img_path = nextnext + imgs
                img = cv2.imread(img_path)
                #print(img[1,1])
                #if img.shape[0]>900:
                row = img.shape[0]
                col = img.shape[1]
                if img[0,0][0]==0 and img[row-2, col-2][0]==0 and img[int(row/2), int(col/2)][0]==0:
                #if img[0,0][0]==0 and img[row-2, col-2][0]==0 :
                    #print(img_path)
                    continue
                else:
                    dest = 'G:\DDSM\CBIS-DDSM\JPEG_test\Calc_trans/' + file + '.jpg'
                    cv2.imwrite(dest, img)
print(count)
            
        
        
