"""
    separate benign and malignant case and generate different csv
    Args:
        path (str):
"""
import csv
import os

read_benign_path = r'D:\PycharmProjects\detectron2\training_data_csv\mammo_dataset\balance_data_csv/train.csv'
write_benign_path = r'D:\PycharmProjects\detectron2\training_data_csv\mammo_dataset\balance_data_csv\benign_data_csv/train.csv'
# separate img in malignant folder and benign folder
rd_img_path = r'D:\Mammograph\ROI_training_dataset\0_benign_img/'
mv_img_path = r'D:\Mammograph\ROI_training_dataset\0_malignant_img/'

read_malignant_path = r'D:\PycharmProjects\detectron2\training_data_csv\mammo_dataset\balance_data_csv/train.csv'
write_malignant_path = r'D:\PycharmProjects\detectron2\training_data_csv\mammo_dataset\balance_data_csv\malignant_data_csv/train.csv'

malignant_list = [1, 4, 7, 14, 19, 38, 40, 41, 49, 54, 60, 68, 70, 73, 79, 98, 99, 102, 107, 109, 112, 124, 126, 128, 129, 136, 139, 146, 158, 159,
                  160, 162, 169, 172, 173, 177, 184, 186, 187, 194, 200, 202, 214, 216, 233, 258, 262, 268, 269, 273, 289, 294, 295, 304, 307, 315,
                  327, 335, 337, 342, 356, 379, 381, 392, 393, 401, 404, 409, 411, 423, 425, 429, 435, 437, 439, 444, 447, 458, 462, 463]
malignant_pick = []

# separate img in malignant folder and benign folder
images = os.listdir(rd_img_path)
for image in images:
    id = int(image.split('_')[1].split('.')[0])
    if id in malignant_list:
        current_path = rd_img_path + image
        new_path = mv_img_path + image
        os.rename(current_path, new_path)
print()

with open(read_malignant_path, newline='') as csvfile:
    rows = csv.reader(csvfile)
    for row in rows:
        id = int(row[0].split('\\')[-1].split('_')[1])
        if (id in malignant_list):
            malignant_pick.append(row[0])
with open(write_malignant_path, 'w', newline='') as csvfile:
    for pick in malignant_pick:
        writer = csv.writer(csvfile)
        writer.writerow([pick])

benign_pick = []
with open(read_benign_path, newline='') as csvfile:
    rows = csv.reader(csvfile)
    for row in rows:
        id = int(row[0].split('\\')[-1].split('_')[1])
        if (id not in malignant_list):
            benign_pick.append(row[0])
with open(write_benign_path, 'w', newline='') as csvfile:
    for pick in benign_pick:
        writer = csv.writer(csvfile)
        writer.writerow([pick])