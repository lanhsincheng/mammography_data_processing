
import os

rd_path = r'G:\DDSM\CBIS-DDSM\JPEG_test\Mass_trans/'
rd_label_path = r'G:\DDSM\CBIS-DDSM/DDSM_masstest_patho.txt'
wb_path = r'G:\project_yolo\project_mammo\training_dataset\images/'

fp = open(rd_label_path, "r")
label_list = []
count = 0
file_no = 378
img_list = os.listdir(rd_path)

while (count<file_no):
    count += 1
    line = fp.readline()
    line = line.split('\n')[0]
    label_list.append(int(line))

i=0
for img in img_list:
    img_name = img.split('.')[0]
    final_wb_path = wb_path + img_name + '.txt'
    fa = open(final_wb_path, "a")
    context = str(label_list[i]) + ' ' + '0.5 0.5 1.0 1.0'
    fa.write(context)
    i += 1
print(i)


