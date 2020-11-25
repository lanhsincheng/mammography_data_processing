# -*- coding: utf-8 -*-
from utils.config import *
import os
import cv2
import xml.etree.cElementTree as ET

# 圖檔路徑
path = LABEL_IMAGE_PATH
high_res_img_path = DCM_TO_JPG_FOLDER_IMAGE
# 存檔路徑
save_path = DEBUG_JPG_PATH
save_resized_path = JPEGImages_PATH
save_crop_ROI_path = CROP_ROI_PATH
cross_matching_path = CROSS_MATCH_PATH
# 存xml檔路徑
annotation_path = Annotations_PATH
roi_annotation_path = ROI_Annotation_PATH
# target_width =  2560
# target_height = 3328

all_label_dirs = os.listdir(path)
fine_img_dirs = os.listdir(high_res_img_path)
get_shape_error_list = []
dcm_to_img_error_list = []
crop_img_error_list = []
def _get_bndbox_coor(jpg_Label_file, folder, type_name):
    """
        Returns start_point, end_point of the labeled_image's bounding box
        Args:
            jpg_Label_file (str): e.g., "D:\Mammograph\original_data\all_labeled_image"
        Returns:
            str: start_point, end_point of the labeled_image's bounding box
            int: ratio_x, ratio_y, resized_crop_img, target_width, target_height (some info of original img )
    """
    # 讀取圖檔
    img = cv2.imread(jpg_Label_file, cv2.IMREAD_UNCHANGED)
    img2 = img.copy()
    rows = img.shape[0]
    cols = img.shape[1]
    #讀取dcm image shape
    if type_name=='CC':
        type_ID = '0001'
    else:
        type_ID = '0002'
    map_path = high_res_img_path + folder + '//' + 'IMG-' + type_ID + '-00001' + '.jpg'
    valid_path = os.path.exists(map_path)
    if valid_path ==False:
        get_shape_error_list.append(folder)
        print('get_shape_error_list : ',get_shape_error_list)
    else:
        dcm_img = cv2.imread(map_path, cv2.IMREAD_UNCHANGED)
        target_width = dcm_img.shape[1]
        target_height = dcm_img.shape[0]
        print('img shape: ', map_path, ' : ', dcm_img.shape)
    coor_list = []
    crop_list = []
    # 擷取影像，去掉周圍的黑邊，此步驟不一定需要
    # for i in range(rows):
    #     for j in range(cols):
    #         if(img[i,j,0]>0 and img[i,j,1]>0 and img[i,j,2]>0 and not (img[i,j,0]==255 and img[i,j,1]==255 and img[i,j,2]==255) and not img[i,j,0]==255 and not img[i,j,1]==255 and not img[i,j,2]==255):
    #             crop_list.append((j,i))

    # print('crop_coor: ' ,crop_list[0], crop_list[-1]) #(446, 22) (979, 714)

    crop_img = img

    # crop_img = img2[ crop_list[0][1]:crop_list[-1][1], crop_list[0][0]:crop_list[-1][0]]
    cv2.imwrite(r'D:\Mammograph\original_data\crop.jpg', crop_img)
    # resize img 至原始dicom大小
    dim = (target_width, target_height) # 2002 2776

    resized_crop_img = cv2.resize(crop_img, dim, interpolation = cv2.INTER_AREA)

    # cv2.imwrite(r'D:\Mammograph\original_data\resize.jpg',resized_crop_img)
    # 讀取原始img上的bounding box
    crop_rows = resized_crop_img.shape[0]
    crop_cols = resized_crop_img.shape[1]
    ratio_x = crop_rows / target_height
    ratio_y = crop_cols / target_width

    for i in range(crop_rows):
        for j in range(crop_cols):
            if(resized_crop_img[i,j,1]>200 and resized_crop_img[i,j,0]<20 and resized_crop_img[i,j,2]<20):
            # if (resized_crop_img[i, j, 1] == 255 and resized_crop_img[i, j, 0] ==0 and resized_crop_img[i, j, 2] ==0):
            # if (resized_crop_img[i, j, 1] == 255 ):
                # print([i,j])
                coor_list.append((j,i))
    # print('box_coor_in_crop_img: ', coor_list[0], coor_list[-1]) #(29, 1982) (599, 2268)
    start_point = coor_list[0]
    end_point = coor_list[-1]
    # color = (0, 0, 255)
    # thickness = 3
    # resized_crop_img = cv2.rectangle(resized_crop_img, start_point, end_point, color, thickness)
    # print(resized_crop_img.shape)
    # cv2.imwrite('check2.jpg', resized_crop_img)
    return start_point, end_point, ratio_x, ratio_y, resized_crop_img, target_width, target_height

def branch1_crop_ROI(left_top, right_bottom, folder, type_name): #Case_100_MLO
    """
            (debug)check if bounding box's position is correct
            Args:
                left_top, right_bottom (str) : coordinate of bounding box
                folder (str) : which Case image belongs to  e.g, "99"
                type_name (str) : what type(view) is the image  e.g, "CC" or "MLO" or "LM" etc.
            Returns:
                no returns
                write back to debug file to check if bounding box's position is correct
                filename  e.g, "Case_99_CC.jpg" or "Case_99_CC.png"
    """

    # print(left_top, ' ', right_bottom, ' ',folder, ' ',type_name)
    if type_name=='CC':
        type_ID = '0001'
    else:
        type_ID = '0002'
    map_path = high_res_img_path + folder + '//' + 'IMG-' + type_ID + '-00001' + '.jpg'
    valid_path = os.path.exists(map_path)
    if valid_path ==False:
        dcm_to_img_error_list.append(folder)
        print('dcm_to_img_error_list : ',dcm_to_img_error_list)
    else:
        img = cv2.imread(map_path, cv2.IMREAD_UNCHANGED)
        # here to define write back path for original dcm_to_jpg image and image type, .jpg or .png
        wb_name_to_original = save_resized_path + 'Case_' + str(folder) + '_' + type_name + '.jpg'
        print('wb_name_to_original : ', wb_name_to_original)
        cv2.imwrite(wb_name_to_original, img)
        color = (0, 0, 255)
        thickness = 3
        with_bndbox_img = cv2.rectangle(img, left_top, right_bottom, color, thickness)
        # here to define write back path and image type, .jpg or .png
        wb_name = save_path + 'Case_' + str(folder) + '_' + type_name + '.jpg'
        print('debug image path(to check bndbox position)',' : ', wb_name)
        cv2.imwrite(wb_name, with_bndbox_img)

def branch1_real_crop(left_top, right_bottom, folder, type_name):
    """
            crop ROI images and write back
            Args:
                left_top, right_bottom (str) : coordinate of bounding box
                folder (str) : which Case image belongs to  e.g, "99"
                type_name (str) : what type(view) is the image  e.g, "CC" or "MLO" or "LM" etc.
            Returns:
                no returns
                write back to crop_ROI file
                filename  e.g, "Case_99_CC.jpg" or "Case_99_CC.png"
    """
    print('start to crop ROI.....')
    if type_name == 'CC':
        type_ID = '0001'
    else:
        type_ID = '0002'
    map_path = high_res_img_path + folder + '//' + 'IMG-' + type_ID + '-00001' + '.jpg'
    valid_path = os.path.exists(map_path)
    if valid_path == False:
        crop_img_error_list.append(folder)
        print('crop_img_error_list : ', crop_img_error_list)
    else:
        img = cv2.imread(map_path, cv2.IMREAD_UNCHANGED)
        # cv2.imshow('mlo',img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # print(left_top, right_bottom)
        #這裡好怪，有時ERROR，可是明明不該相反的，第一個才是對的才對
        crop_img = img[left_top[1]:right_bottom[1], left_top[0]:right_bottom[0]]
        # crop_img = img[left_top[1]:right_bottom[1], right_bottom[0]:left_top[0]]
        # cv2.imshow('mlo',crop_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        print(left_top, right_bottom)
        # here to define write back path and image type, .jpg or .png
        wb_name = save_crop_ROI_path + 'Case_' + str(folder) + '_' + type_name + '.jpg'
        print('crop image path(only cropped part)',' : ',wb_name)
        cv2.imwrite(wb_name, crop_img)

def branch2_xml(left_top, right_bottom, jpg_Label_file, folder_num, type_name, twidth, theight):
    """
            create Annotations for training process
            Args:
                left_top, right_bottom (str) : coordinate of bounding box
                folder (str) : which Case image belongs to  e.g, "99"
                type_name (str) : what type(view) is the image  e.g, "CC" or "MLO" or "LM" etc.
            Returns:
                no returns
                write back xml file with bounding box info as labeled_images
                filename  e.g, "Case_99_CC.xml"
    """
    annotation = ET.Element('annotation')
    folder = ET.SubElement(annotation, 'folder')
    folder.text = folder_num
    filename = ET.SubElement(annotation, 'filename')
    filename.text = jpg_Label_file
    path = ET.SubElement(annotation, 'path')
    path.text = save_path + jpg_Label_file  #'D:\Mammograph\final_jpg_debug/'
    source = ET.SubElement(annotation, 'source')
    database = ET.SubElement(source, 'database')
    database.text = 'Unknown'
    size = ET.SubElement(annotation, 'size')
    width = ET.SubElement(size, 'width')
    width.text = str(twidth) #'2560'
    height = ET.SubElement(size, 'height')
    height.text = str(theight) #'3328'
    depth = ET.SubElement(size, 'depth')
    depth.text = '3'
    segmented = ET.SubElement(annotation, 'segmented')
    segmented.text = '0'
    object = ET.SubElement(annotation, 'object')
    name = ET.SubElement(object, 'name')
    name.text = 'malignant'
    pose = ET.SubElement(object, 'pose')
    pose.text = 'Unspecified'
    truncated = ET.SubElement(object, 'truncated')
    truncated.text = '0'
    difficult = ET.SubElement(object, 'difficult')
    difficult.text = '0'
    bndbox = ET.SubElement(object, 'bndbox')
    xmin = ET.SubElement(bndbox, 'xmin')
    xmin.text = str(left_top[0])
    ymin = ET.SubElement(bndbox, 'ymin')
    ymin.text = str(left_top[1])
    xmax = ET.SubElement(bndbox, 'xmax')
    xmax.text = str(right_bottom[0])
    ymax = ET.SubElement(bndbox, 'ymax')
    ymax.text = str(right_bottom[1])

    # mydata = ET.tostring(annotation, encoding="utf-8")
    # print(type(mydata))
    wb_xml_path = annotation_path + jpg_Label_file.split(".")[0] +'.xml'
    # print(wb_xml_path)
    myfile = open(wb_xml_path, "w")
    # myfile.write(mydata)
    myfile.write(ET.tostring(annotation).decode("utf-8"))

def branch2_real_crop_xml(jpg_Label_file, folder_num, type_name):
    """
                branch2_real_crop_xml(jpg_Label_file, folder, type_name, width, height)
                create Annotations for roi training process
                Args:
                    folder_num (str) : which Case image belongs to  e.g, "99"
                    type_name (str) : what type(view) is the image  e.g, "CC" or "MLO" or "LM" etc.
                    twidth (int) / theight (int) : width / height of the crop roi image
                Returns:
                    no returns
                    write back xml file with roi info
                    filename  e.g, "Case_99_CC.xml"
        """
    annotation = ET.Element('annotation')
    folder = ET.SubElement(annotation, 'folder')
    folder.text = folder_num
    filename = ET.SubElement(annotation, 'filename')
    filename.text = jpg_Label_file
    path = ET.SubElement(annotation, 'path')
    path.text = save_crop_ROI_path + jpg_Label_file  # 'D:\Mammograph\ROI_training_dataset\JPEGImages/Case name.jpg'

    img = cv2.imread(save_crop_ROI_path + jpg_Label_file, cv2.IMREAD_UNCHANGED)
    theight = img.shape[0]
    twidth = img.shape[1]

    source = ET.SubElement(annotation, 'source')
    database = ET.SubElement(source, 'database')
    database.text = 'Unknown'
    size = ET.SubElement(annotation, 'size')
    width = ET.SubElement(size, 'width')
    width.text = str(twidth)
    height = ET.SubElement(size, 'height')
    height.text = str(theight)
    depth = ET.SubElement(size, 'depth')
    depth.text = '3'
    segmented = ET.SubElement(annotation, 'segmented')
    segmented.text = '0'
    object = ET.SubElement(annotation, 'object')
    name = ET.SubElement(object, 'name')
    # name.text = 'benign'
    name.text = 'malignant'
    pose = ET.SubElement(object, 'pose')
    pose.text = 'Unspecified'
    truncated = ET.SubElement(object, 'truncated')
    truncated.text = '0'
    difficult = ET.SubElement(object, 'difficult')
    difficult.text = '0'
    bndbox = ET.SubElement(object, 'bndbox')
    xmin = ET.SubElement(bndbox, 'xmin')
    xmin.text = '0'
    ymin = ET.SubElement(bndbox, 'ymin')
    ymin.text = '0'
    xmax = ET.SubElement(bndbox, 'xmax')
    xmax.text = str(twidth)
    ymax = ET.SubElement(bndbox, 'ymax')
    ymax.text = str(theight)

    # mydata = ET.tostring(annotation, encoding="utf-8")
    # print(type(mydata))
    wb_xml_path = roi_annotation_path + jpg_Label_file.split(".")[0] + '.xml'
    # print(wb_xml_path)
    myfile = open(wb_xml_path, "w")
    # myfile.write(mydata)
    myfile.write(ET.tostring(annotation).decode("utf-8"))

def branch3_cross_matching(left_top, right_bottom, ratio_x, ratio_y, golden_path, golden_filename, resized_crop_img):

    # golden_img = cv2.imread(golden_path)
    x1 = round(left_top[0]*ratio_x)
    x2 = round(left_top[1]*ratio_y)
    y1 = round(right_bottom[0]*ratio_x)
    y2 = round(right_bottom[1]*ratio_y)
    scale_left_top = (x1, x2)
    scale_right_bottom = (y1, y2)

    color = (0, 0, 255)
    thickness = 3
    with_bndbox_img = cv2.rectangle(resized_crop_img, scale_left_top, scale_right_bottom, color, thickness)
    # here to define write back path and image type, .jpg or .png
    wb_name = cross_matching_path + golden_filename
    print('cross_matching image path(to check bndbox position)', ' : ', wb_name)
    cv2.imwrite(wb_name, with_bndbox_img)

for jpg_Label_file in all_label_dirs:
    midname = jpg_Label_file.split("_")
    folder = midname[1] #99
    type_name = midname[2].split(".")[0] #MLO, CC, LM
    left_top, right_bottom, ratio_x, ratio_y, resized_crop_img, width, height =_get_bndbox_coor(path + jpg_Label_file, folder, type_name)
    branch1_crop_ROI(left_top, right_bottom, folder, type_name)
    branch1_real_crop(left_top, right_bottom, folder, type_name)
    branch2_xml(left_top, right_bottom, jpg_Label_file, folder, type_name, width, height)
    branch2_real_crop_xml(jpg_Label_file, folder, type_name) #Case-1-CC.jpg
    branch3_cross_matching(left_top, right_bottom, ratio_x, ratio_y, path + jpg_Label_file, jpg_Label_file, resized_crop_img)


## report error images
print('dcm_to_img_error_list : ',dcm_to_img_error_list)
print('crop_img_error_list : ',crop_img_error_list)