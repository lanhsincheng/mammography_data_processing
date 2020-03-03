# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 18:30:15 2020

@author: idip.claire
"""

import numpy as np
import cv2
import csv
import os, sys
from os import walk
from os.path import join
from matplotlib import pyplot as plt
import xml.etree.cElementTree as ET

img_path = r'D:\Mammograph\result\faster_rcnn_R50_NO_COCO_PRETRAIN_visualize_0227/'
ori_img_path = r'D:\Mammograph\training_dataset\JPEGImages/'
path = r'D:\Mammograph\training_dataset\Annotations/'
ouput_path = r'D:\Mammograph\answer\faster_rcnn_R50_NO_COCO_PRETRAIN_visualize_0227/'
all_label_dirs = os.listdir(img_path)
image = cv2.imread(img_path)

for img_name in all_label_dirs:

    image = cv2.imread(img_path+img_name)
    image_ori = cv2.imread(ori_img_path+img_name)
    print(img_name)
    print(image.shape)
    print(image_ori.shape)
    image_resize = cv2.resize(image,(image_ori.shape[1],image_ori.shape[0]))
    print(image_resize.shape)
    xml_name=img_name.split('.')[0]
    #print(xml_name)
    tree = ET.parse(path+xml_name+'.xml')
    root = tree.getroot()
    print(root[6][4][0].text,root[6][4][1].text,root[6][4][2].text,root[6][4][3].text)
    cv2.rectangle(image_resize, (int(root[6][4][0].text), int(root[6][4][1].text)), (int(root[6][4][2].text), int(root[6][4][3].text)), (0, 255, 0), 7)
    img_final = cv2.resize(image_resize,(image.shape[1],image.shape[0]))
    print(img_final.shape)
    #print(path+xml_name+'.xml')
    cv2.imwrite(ouput_path+img_name,img_final)


