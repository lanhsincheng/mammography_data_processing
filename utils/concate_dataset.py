"""
        Concate 2 feature img vertically or horizontally
        Usage : change path
        Args:
            data csv_file (str): e.g., "D:\Mammograph\original_data\all_labeled_image.csv"
            data img_folder (str): e.g., "D:\Mammograph\original_data\all_labeled_image"
        Returns:

    """
import cv2
import os
import xml.etree.cElementTree as ET

original_img_dir = r'D:\Mammograph\ROI_training_dataset\JPEGImages/'
gabor0_img_dir = r'D:\Mammograph\gabor_training_dataset\JPEGImages(f=0.1_t=0)/'
gabor1_img_dir = r'D:\Mammograph\fft_training_dataset\JPEGImages/'
gabor2_img_dir = r'D:\Mammograph\ROI_training_dataset\preprocess/'

wb_path_img = r'D:\Mammograph\concate_training_dataset\JPEGImages_seg/'
wb_path_xml = r'D:\Mammograph\concate_training_dataset\Annotations_1/'

original_images = os.listdir(original_img_dir)

malignant_list = [1, 4, 7, 14, 19, 38, 40, 41, 49, 54, 60, 68, 70, 73, 79, 98, 99, 102, 107, 109, 112, 124, 126, 128, 129, 136, 139, 146, 158, 159,
                  160, 162, 169, 172, 173, 177, 184, 186, 187, 194, 200, 202, 214, 216, 233, 258, 262, 268, 269, 273, 289, 294, 295, 304, 307, 315,
                  327, 335, 337, 342, 356, 379, 381, 392, 393, 401, 404, 409, 411, 423, 425, 429, 435, 437, 439, 444, 447, 458, 462, 463]


def concate_img(original_img, gabor0_img, gabor1_img, gabor2_img):
    im0_v = cv2.vconcat([original_img, gabor0_img])
    im1_v = cv2.vconcat([gabor1_img, gabor2_img])
    im_v = cv2.hconcat([im0_v, im1_v])
    cv2.imwrite(wb_path_img + img, im_v)

def gen_annotations(xfolder, clsid, xfilename, xwidth, xheight):
    annotation = ET.Element('annotation')
    folder = ET.SubElement(annotation, 'folder')
    folder.text = str(xfolder)
    filename = ET.SubElement(annotation, 'filename')
    filename.text = xfilename
    path = ET.SubElement(annotation, 'path')
    path.text = wb_path_img + xfilename  # 'D:\Mammograph\ROI_training_dataset\JPEGImages/Case name.jpg'

    source = ET.SubElement(annotation, 'source')
    database = ET.SubElement(source, 'database')
    database.text = 'Unknown'
    size = ET.SubElement(annotation, 'size')
    width = ET.SubElement(size, 'width')
    width.text = str(xwidth)
    height = ET.SubElement(size, 'height')
    height.text = str(xheight)
    depth = ET.SubElement(size, 'depth')
    depth.text = '3'
    segmented = ET.SubElement(annotation, 'segmented')
    segmented.text = '0'
    # 1st obj
    object = ET.SubElement(annotation, 'object')
    name = ET.SubElement(object, 'name')
    name.text = clsid
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
    xmax.text = str(xwidth-1)
    ymax = ET.SubElement(bndbox, 'ymax')
    ymax.text = str(xheight-1)
    # # 2nd obj
    # object = ET.SubElement(annotation, 'object')
    # name = ET.SubElement(object, 'name')
    # second_clsid = clsid + '0gabor'
    # name.text = second_clsid
    # pose = ET.SubElement(object, 'pose')
    # pose.text = 'Unspecified'
    # truncated = ET.SubElement(object, 'truncated')
    # truncated.text = '0'
    # difficult = ET.SubElement(object, 'difficult')
    # difficult.text = '0'
    # bndbox = ET.SubElement(object, 'bndbox')
    # xmin = ET.SubElement(bndbox, 'xmin')
    # xmin.text = '0'
    # ymin = ET.SubElement(bndbox, 'ymin')
    # ymin.text = str(xheight-1)
    # xmax = ET.SubElement(bndbox, 'xmax')
    # xmax.text = str(xwidth-1)
    # ymax = ET.SubElement(bndbox, 'ymax')
    # ymax.text = str(xheight * 2-1)
    # # 3rd obj
    # object = ET.SubElement(annotation, 'object')
    # name = ET.SubElement(object, 'name')
    # second_clsid = clsid + '1gabor'
    # name.text = second_clsid
    # pose = ET.SubElement(object, 'pose')
    # pose.text = 'Unspecified'
    # truncated = ET.SubElement(object, 'truncated')
    # truncated.text = '0'
    # difficult = ET.SubElement(object, 'difficult')
    # difficult.text = '0'
    # bndbox = ET.SubElement(object, 'bndbox')
    # xmin = ET.SubElement(bndbox, 'xmin')
    # xmin.text = str(xwidth - 1)
    # ymin = ET.SubElement(bndbox, 'ymin')
    # ymin.text = '0'
    # xmax = ET.SubElement(bndbox, 'xmax')
    # xmax.text = str(xwidth * 2 - 1)
    # ymax = ET.SubElement(bndbox, 'ymax')
    # ymax.text = str(xheight - 1)
    # # 4th obj
    # object = ET.SubElement(annotation, 'object')
    # name = ET.SubElement(object, 'name')
    # second_clsid = clsid + '2gabor'
    # name.text = second_clsid
    # pose = ET.SubElement(object, 'pose')
    # pose.text = 'Unspecified'
    # truncated = ET.SubElement(object, 'truncated')
    # truncated.text = '0'
    # difficult = ET.SubElement(object, 'difficult')
    # difficult.text = '0'
    # bndbox = ET.SubElement(object, 'bndbox')
    # xmin = ET.SubElement(bndbox, 'xmin')
    # xmin.text = str(xwidth - 1)
    # ymin = ET.SubElement(bndbox, 'ymin')
    # ymin.text = str(xheight - 1)
    # xmax = ET.SubElement(bndbox, 'xmax')
    # xmax.text = str(xwidth * 2 - 1)
    # ymax = ET.SubElement(bndbox, 'ymax')
    # ymax.text = str(xheight * 2 - 1)

    # mydata = ET.tostring(annotation, encoding="utf-8")
    # print(type(mydata))
    wb_xml_path = wb_path_xml + xfilename.split(".")[0] + '.xml'
    # print(wb_xml_path)
    myfile = open(wb_xml_path, "w")
    # myfile.write(mydata)
    myfile.write(ET.tostring(annotation).decode("utf-8"))

## task1 : concate img
for img in original_images:
    original_img = cv2.imread(original_img_dir + img)
    gabor0_img = cv2.imread(gabor0_img_dir + img)
    gabor1_img = cv2.imread(gabor1_img_dir + img)
    gabor2_img = cv2.imread(gabor2_img_dir + img)

    ## you can disable this function if you don't need
    concate_img(original_img, gabor0_img, gabor1_img, gabor2_img)

    ## task2 : generate 3 annotation each concated img
    # xwidth = original_img.shape[1]
    # xheight = original_img.shape[0]
    # id = int(img.split('.')[0].split('_')[1])
    # if id in malignant_list:
    #     clsid = 'malignant'
    # else:
    #     clsid = 'benign'
    # gen_annotations(id, clsid, img, xwidth, xheight)