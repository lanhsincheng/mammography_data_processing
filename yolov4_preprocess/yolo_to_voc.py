"""
    parse yolo txt into voc annotations
    Usage :
    Args:
        path (str):
"""
import glob
import os
import cv2
import xml.etree.cElementTree as ET
import fnmatch
import csv

rd_path = r'D:\Mammograph\mammoDDSM/'
wb_path1 = r'D:\Mammograph\mammoDDSM\malignant/'
wb_path2 = r'D:\Mammograph\mammoDDSM\benign/'

rd_path_testlist = r'D:\Mammograph\mammoDDSM/testlist.txt'
rd_path_xml = r'D:\Mammograph\DDSM_training_dataset\Annotations'
wb_path_testtxt = r'D:\Mammograph\DDSM_training_dataset/test.txt'
wb_path_traintxt = r'D:\Mammograph\DDSM_training_dataset/train.txt'

def parse_txt(rd_path):
    img_list = []
    id_list = []
    os.chdir(rd_path)
    for txt_file in glob.glob("*.txt"):
        f = open(rd_path + txt_file, "r")
        id = f.readline().split(" ")[0]
        img_list.append(txt_file.split(".")[0] + '.jpg')
        id_list.append(id)
    return img_list, id_list

def gen_xml(folder_id, img, wb_path_option, swidth, sheight, id):
    annotation = ET.Element('annotation')
    folder = ET.SubElement(annotation, 'folder')
    folder.text = folder_id
    filename = ET.SubElement(annotation, 'filename')
    filename.text = img
    path = ET.SubElement(annotation, 'path')
    path.text = wb_path_option + img
    source = ET.SubElement(annotation, 'source')
    database = ET.SubElement(source, 'database')
    database.text = 'Unknown'
    size = ET.SubElement(annotation, 'size')
    width = ET.SubElement(size, 'width')
    width.text = str(swidth)
    height = ET.SubElement(size, 'height')
    height.text = str(sheight)
    depth = ET.SubElement(size, 'depth')
    depth.text = '3'
    segmented = ET.SubElement(annotation, 'segmented')
    segmented.text = '0'
    object = ET.SubElement(annotation, 'object')
    name = ET.SubElement(object, 'name')
    name.text = str(id)
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
    xmax.text = str(swidth-1)
    ymax = ET.SubElement(bndbox, 'ymax')
    ymax.text = str(sheight-1)

    wb_xml_path = wb_path_option + img.split(".")[0] + '.xml'
    myfile = open(wb_xml_path, "w")
    myfile.write(ET.tostring(annotation).decode("utf-8"))


# # dir contain .jpg/.txt
# # read all txt and parse, finally separate each class images and generated-xml to different class dir
# img_list, id_list= parse_txt(rd_path)
# for img, id in zip(img_list, id_list):
#     image = cv2.imread(rd_path + img)
#     width = image.shape[1]
#     height = image.shape[0]
#     if (id == '0'):
#         gen_xml('malignant', img, wb_path1, width, height, 0) # folder_id, img, wb_path_option, swidth, sheight, id
#         cv2.imwrite(wb_path1 + img, image)
#     else:
#         gen_xml('benign', img, wb_path2, width, height, 1)
#         cv2.imwrite(wb_path2 + img, image)

# generate test.txt/train.txt
fp = open(rd_path_testlist, 'r')
files = os.listdir(rd_path_xml)
for i in range(151): # 151 lines
    patient_id = fp.readline().split("\n")[0]
    pattern = patient_id + '_*'
    for name in files:
        if (fnmatch.fnmatch(name, pattern)):
            fa = open(wb_path_testtxt, "a")
            fa.write(name.split(".")[0] + '.jpg' + '\n')
        else:
            fa = open(wb_path_traintxt, "a")
            context = 'D:/' + name.split(".")[0] + '.jpg'
            fa.write(name.split(".")[0] + '.jpg' + '\n')
