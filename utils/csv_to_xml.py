"""
        #####
        deection model as classification model
        #####

        Convert csv with image name and label info to corresponding xml
        Usage : change csv file path and img folder path
        Args:
            data csv_file (str): e.g., "D:\PycharmProjects\detectron2\training_data_csv\mango_dataset/test.csv"
            data img_folder (str): e.g., "D:\AIMango\training_dataset\JPEGImages/"
        Returns:
            each xml file recording in one csv file
    """
import os
import csv
import cv2
import xml.etree.cElementTree as ET

# csv path
csv_path = r'D:\PycharmProjects\detectron2\training_data_csv\mango_dataset/test.csv'
# img path
img_path = r'D:\AIMango\training_dataset\JPEGImages/'
# write back xml
xml_path = r'D:\AIMango\training_dataset\Annotations/'

def parsing_csv():
    file_name_list = []
    class_id_list = []
    with open(csv_path, newline='') as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            # print(row) # ['00002.jpg', 'C']
            file_name = row[0]
            class_id = row[1]
            file_name_list.append(file_name)
            class_id_list.append(class_id)

    return file_name_list, class_id_list

def generate_xml(file_name, class_id):

    """
            create Annotations for training process
            Args:

            Returns:
                no returns
                write back xml file with roi info
                filename  e.g, "00002.xml"
    """
    annotation = ET.Element('annotation')
    folder = ET.SubElement(annotation, 'folder')
    folder.text = 'AIMango'
    filename = ET.SubElement(annotation, 'filename')
    filename.text = file_name
    path = ET.SubElement(annotation, 'path')
    path.text = img_path + file_name  # 'D:\AIMango\training_dataset\JPEGImages/00002.jpg'

    img = cv2.imread(img_path + file_name, cv2.IMREAD_UNCHANGED)
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
    name.text = class_id
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

    wb_xml_path = xml_path + file_name.split(".")[0] + '.xml'
    myfile = open(wb_xml_path, "w")
    myfile.write(ET.tostring(annotation).decode("utf-8"))

file_name_list, class_id_list = parsing_csv()
for file_name, class_id in zip(file_name_list, class_id_list):
    generate_xml(file_name, class_id)
