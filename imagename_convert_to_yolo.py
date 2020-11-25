"""
    rename image as case attached with its corresponding label
    Args:
        path (str):
"""
import os
import cv2

read_image_path = r'D:\Mammograph\YOLO_ROI_training_dataset\YOLOImages\gabor(f=1.6_t=0)\test/'
write_image_path = r'D:\Mammograph\YOLO_ROI_training_dataset\YOLOImages\gabor(f=1.6_t=0)\new_test/'


malignant_list = [1, 4, 7, 14, 19, 38, 40, 41, 49, 54, 60, 68, 70, 73, 79, 98, 99, 102, 107, 109, 112, 124, 126, 128, 129, 136, 139, 146, 158, 159,
                  160, 162, 169, 172, 173, 177, 184, 186, 187, 194, 200, 202, 214, 216, 233, 258, 262, 268, 269, 273, 289, 294, 295, 304, 307, 315,
                  327, 335, 337, 342, 356, 379, 381, 392, 393, 401, 404, 409, 411, 423, 425, 429, 435, 437, 439, 444, 447, 458, 462, 463]


def num2str(img_name):
    if (img_name.split("_")[3] == 'CC.jpg'):
        img_name = img_name[7:-7]
    else:
        img_name = img_name[7:-8]
    new_str = ''
    for name_str in img_name:
        name_str = int(name_str) + 65
        new_str = new_str + chr(name_str)

    return new_str

count = 0
images = os.listdir(read_image_path)
for image in images:
    img = cv2.imread(read_image_path + image)
    id = int(image.split("_")[2])
    view = image.split("_")[3]
    if (id in malignant_list):
        if(view == 'CC.jpg'):
            rename = '0_Case_' + num2str(image) + '_CC.jpg'
        else:
            rename = '0_Case_' + num2str(image) + '_MLO.jpg'
    else:
        if (view == 'CC.jpg'):
            rename = '1_Case_' + num2str(image) + '_CC.jpg'
        else:
            rename = '1_Case_' + num2str(image) + '_MLO.jpg'

    wb_path = write_image_path + rename
    if os.path.exists(wb_path):
        print(wb_path)
    cv2.imwrite(wb_path, img)
    count += 1
    # print(count)