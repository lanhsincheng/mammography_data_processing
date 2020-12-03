import numpy as np
from PIL import Image
import os, sys
from os import path
import cv2
import xml.etree.cElementTree as ET


# 存xml檔路徑
annotation_path = r'D:\Mammograph\training_dataset\tt/'
wb =  r'D:\Mammograph\training_dataset\Annotations/'

xml_dirs = os.listdir(annotation_path)
scale_size_list = []

def parse(filename, name):
    tree = ET.parse(filename)
    read_shape_path = tree.find("path").text
    print(read_shape_path)
    img = cv2.imread(read_shape_path)
    theight = img.shape[0]
    twidth = img.shape[1]
    
    for obj in tree.findall("size"):
         if obj.find("width").text!=str(twidth):
             obj.find("width").text = str(twidth)
         if obj.find("height").text!=str(theight):
             obj.find("height").text = str(theight)
             scale_size_list.append(filename)
    wb_xml_path = wb + name.split(".")[0] + '.xml'
    tree.write(wb_xml_path)
    
    
for filename in xml_dirs:
    path_to_xml = annotation_path + filename
    parse(path_to_xml, filename)
print(scale_size_list)
