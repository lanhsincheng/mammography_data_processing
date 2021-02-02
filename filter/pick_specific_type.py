"""
        pick file name which are specified in pick.txt
        Usage : use with pick.txt
        Args:

        Returns:
"""
import os
import cv2
rd_path = r'G:\DDSM\CBIS-DDSM\JPEG_train\Calc_trans/'
wb_path = r'G:\DDSM\CBIS-DDSM\JPEG_val/'
# if want to filter
filter_path = r'G:\project_yolo\project_mammo\training_dataset\test/'

pick_list=[]
fp = open('pick.txt', "r")
while (1):
    line = fp.readline().split('\n')[0]
    pick_list.append(line)
    if line =="":
        break
    else: continue

img_list = os.listdir(rd_path)
i = 0
k = 0
#print(pick_list)
# for img in img_list:
for img in pick_list:
    rd_final_path = rd_path + img + '.jpg'
    pic = cv2.imread(rd_final_path)
    img_name = img.split('.')[0]
    final_wb_path = wb_path + img_name + '.jpg'
    cv2.imwrite(final_wb_path, pic)
    k+=1

    # if want to filter
    # check_path = filter_path + img_name + '.jpg'
#     if img_name in pick_list and not os.path.isfile(check_path):
#         cv2.imwrite(final_wb_path,pic )
#         print(final_wb_path)
#         i += 1
#     else: k+=1
# print(i, k)
print(k)