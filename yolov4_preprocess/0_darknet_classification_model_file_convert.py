"""
    data file name should include custom class_id. eg. mal_Calc-Test_P_00038_LEFT_CC_1.jpg
    Usage : 0: malignant ; 1: benign
    Args:
        rd_path (str): ./images/
        rd_txt_path (str): ./*.txt 's mother directory
        wb_path (str) : ./images/
    Returns :

"""
import os
import cv2

rd_path = r'G:\DDSM\CBIS-DDSM\JPEG_train\Calc_trans/'
rd_txt_path = r'G:\project_yolo\0911_DDSM_original\training_dataset\txt/'
wb_path = r'G:\project_yolo\project_mammo\training_dataset\train/'


txts = os.listdir(rd_txt_path)
count = 0
for txt in txts:
    img = cv2.imread(rd_path + txt.split(".")[0] + '.jpg')
    fp = open(rd_txt_path + txt, "r")


    line = fp.readline()
    class_id = line.split(" ")[0]
    if(int(class_id) == 0):
        wb_final_path = wb_path + 'malignant_' + txt.split(".")[0] + '.jpg'
        cv2.imwrite(wb_final_path, img)
    else:
        wb_final_path = wb_path + 'benign_' + txt.split(".")[0] + '.jpg'
        cv2.imwrite(wb_final_path, img)
    count += 1
    print(count)
