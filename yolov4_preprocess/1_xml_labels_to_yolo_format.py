"""
    data label in xml format to yolo txt format
    Usage :
    Args:
        rd_xml_path (str): ./Annotations/*.xml/
        wb_path (str) : ./txt/
    Returns :
        *.txt file with context : 1 0.5 0.5 1.0 1.0 for corresponding xml cases
"""
import os
import xml.etree.cElementTree as ET


rd_xml_path = r'D:\Mammograph\ROI_training_dataset\Annotations-test/'
wb_path = r'D:\Mammograph\ROI_training_dataset\txt/'
check_path = r'D:\Mammograph\ROI_training_dataset\txt/'

def read_xml(xml_path):
    root = ET.parse(xml_path).getroot()
    for label in root.findall('object/name'):
        label_val = label.text

    return label_val

def write_txt(image, label_val):
    txt_name = wb_path + image + '.txt'
    fa = open(txt_name, "a")
    if label_val=='malignant':
        label_id = 0
    else:
        label_id = 1
    line =str(label_id) + ' ' + '0.5 0.5 1.0 1.0'
    fa.write(line)

def fetch_label(check_path, image):
    final_check_path = check_path + image + '.txt'
    ff = open(final_check_path, "r")
    line = ff.readline().split(' ')[0]

    return line
ben = 0
mal = 0
xmls = os.listdir(rd_xml_path)
for xml in xmls:
    xml_path = rd_xml_path + xml
    image = xml.split('.')[0]
    label_val = read_xml(xml_path)
    write_txt(image, label_val)
    classid = fetch_label(check_path, image)
    if classid == '0':
        mal += 1
    elif classid == '1':
        ben += 1
print('mal count: ', mal)
print('ben count: ', ben)