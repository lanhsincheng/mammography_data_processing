"""
    (ROI) create xml format for whole patho, type and distribution.
    And automatic get the filter images by filter_filr.py.
    Usage :
    Args:
        path (str):

"""
import os
import cv2
import xml.etree.cElementTree as ET



# img path
rd_path = r'G:\DDSM\CBIS-DDSM\JPEG_test\Calc_trans/'
# lable txt path
label_rd_path = r'G:\DDSM\CBIS-DDSM/DDSM_calctest_patho.txt'
# type txt path
type_rd_path = r'G:\DDSM\CBIS-DDSM/DDSM_calctest_type.txt'
# distribution txt path
distribution_rd_path = r'G:\DDSM\CBIS-DDSM/DDSM_calctest_distribution.txt'

wb_path = r'G:\DDSM\CBIS-DDSM\Annotation_test/'

def get_base_info(image, image_rd_path):
    img = cv2.imread(image_rd_path)
    rows = img.shape[0]
    cols = img.shape[1]
    filename = image_rd_path
    folder = image

    return rows, cols, filename, folder

def get_label(label_rd_path):
    label_list = []
    fp = open(label_rd_path, 'r')
    while(1):
        line = fp.readline()
        if not line: break
        label_list.append(line.split('\n')[0])
    print(len(label_list))
    return label_list

def get_type(type_rd_path):
    type_list = []
    fp = open(type_rd_path, 'r')
    while (1):
        line = fp.readline()
        if not line: break
        type_list.append(line.split('\n')[0])
    print(len(type_list))
    return type_list

def get_distribution(distribution_rd_path):
    distribution_list = []
    fp = open(distribution_rd_path, 'r')
    while (1):
        line = fp.readline()
        if not line: break
        distribution_list.append(line.split('\n')[0])
    print(len(distribution_list))
    return distribution_list

def gen_xml(image, theight, twidth, file_name, folder_name, label, type, distribution):

    annotation = ET.Element('annotation')
    folder = ET.SubElement(annotation, 'folder')
    folder.text = folder_name
    filename = ET.SubElement(annotation, 'filename')
    filename.text = image
    path = ET.SubElement(annotation, 'path')
    path.text = file_name

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
    name.text = label
    pose = ET.SubElement(object, 'pose')
    pose.text = type
    truncated = ET.SubElement(object, 'truncated')
    truncated.text = distribution
    difficult = ET.SubElement(object, 'difficult')
    difficult.text = '0'
    bndbox = ET.SubElement(object, 'bndbox')
    xmin = ET.SubElement(bndbox, 'xmin')
    xmin.text = '0'
    ymin = ET.SubElement(bndbox, 'ymin')
    ymin.text = '0'
    xmax = ET.SubElement(bndbox, 'xmax')
    xmax.text = str(twidth-1)
    ymax = ET.SubElement(bndbox, 'ymax')
    ymax.text = str(theight-1)

    wb_xml_path = wb_path + image.split(".")[0] + '.xml'
    myfile = open(wb_xml_path, "w")
    myfile.write(ET.tostring(annotation).decode("utf-8"))


images = os.listdir(rd_path)
# name
label_list = get_label(label_rd_path)
# pose
type_list = get_type(type_rd_path)
# truncated
distribution_list = get_distribution(distribution_rd_path)

for image, label, type, distribution in zip(images, label_list, type_list, distribution_list):
    image_rd_path = rd_path + image
    height, width, filename, folder = get_base_info(image, image_rd_path)
    gen_xml(image, height, width, filename, folder, label, type, distribution)




