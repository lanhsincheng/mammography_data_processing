"""
    (ROI) And automatic get the filter images.
    Usage :
    Args:
        path (str):
"""
import os
import cv2
import xml.etree.cElementTree as ET

# contain *.xml
rd_path = r'G:\DDSM\CBIS-DDSM\Annotation_test/'
wb_path = r'G:\DDSM\CBIS-DDSM\JPEG_test\0_filter\0_Calc_PUNCTATE/'

filter_typr_list = ['label', 'type', 'distribution']
filter_type = filter_typr_list[1]

# DDSM
# option = ['AMORPHOUS', 'AMORPHOUS-PLEOMORPHIC', 'AMORPHOUS-ROUND_AND_REGULAR']
# option = ['COARSE', 'COARSE-ROUND_AND_REGULAR-LUCENT_CENTERED', 'COARSE-PLEOMORPHIC', 'COARSE-ROUND_AND_REGULAR', 'COARSE-ROUND_AND_REGULAR-LUCENT_CENTER', 'COARSE-LUCENT_CENTER']
# option = ['DYSTROPHIC']
# option = ['EGGSHELL']
# option = ['FINE_LINEAR_BRANCHING']
# option = ['LARGE_RODLIKE', 'LARGE_RODLIKE-ROUND_AND_REGULAR']
# option = ['LUCENT_CENTER', 'LUCENT_CENTERED', 'LUCENT_CENTER-PUNCTATE']
# option = ['MILK_OF_CALCIUM']
# option = ['N/A']
# option = ['PLEOMORPHIC', 'PLEOMORPHIC-FINE_LINEAR_BRANCHING', 'PLEOMORPHIC-PLEOMORPHIC', 'PLEOMORPHIC-AMORPHOUS']
#option = ['PUNCTATE', 'PUNCTATE-AMORPHOUS', 'PUNCTATE-FINE_LINEAR_BRANCHING', 'PUNCTATE-LUCENT_CENTER', 'PUNCTATE-PLEOMORPHIC', 'PUNCTATE-ROUND_AND_REGULAR', 'PUNCTATE-AMORPHOUS-PLEOMORPHIC' ]
# option = ['ROUND_AND_REGULAR', 'ROUND_AND_REGULAR-AMORPHOUS', 'ROUND_AND_REGULAR-EGGSHELL', 'ROUND_AND_REGULAR-LUCENT_CENTER', 'ROUND_AND_REGULAR-LUCENT_CENTER-DYSTROPHIC', 'ROUND_AND_REGULAR-LUCENT_CENTERED', 'ROUND_AND_REGULAR-LUCENT_CENTER-PUNCTATE', 'ROUND_AND_REGULAR-PLEOMORPHIC', 'ROUND_AND_REGULAR-PUNCTATE', 'ROUND_AND_REGULAR-PUNCTATE-AMORPHOUS']
# option = ['SKIN', 'SKIN-COARSE-ROUND_AND_REGULAR', 'SKIN-PUNCTATE', 'SKIN-PUNCTATE-ROUND_AND_REGULAR']
# option = ['VASCULAR', 'VASCULAR-COARSE', 'VASCULAR-COARSE-LUCENT_CENTERED', 'VASCULAR-COARSE-LUCENT_CENTER-ROUND_AND_REGULAR-PUNCTATE']

# in-house


def read_xml(xml_path):
    root = ET.parse(xml_path).getroot()
    for path in root.findall('path'):
        path_val = path.text
    for label in root.findall('object/name'):
        label_val = label.text
    for type in root.findall('object/pose'):
        type_val = type.text
    for distribution in root.findall('object/truncated'):
        distribution_val = distribution.text

    return path_val, label_val, type_val, distribution_val

def filtering(image, path_val, label_val, type_val, distribution_val):
    filter_img_wb_path = wb_path + image + '.jpg'
    if filter_type == 'label':
        if label_val in option:
            img = cv2.imread(path_val)
            cv2.imwrite(filter_img_wb_path, img)
    elif filter_type == 'type':
        if type_val in option:
            img = cv2.imread(path_val)
            cv2.imwrite(filter_img_wb_path, img)
    else :
        if distribution_val in option:
            img = cv2.imread(path_val)
            cv2.imwrite(filter_img_wb_path, img)


xmls = os.listdir(rd_path)
for xml in xmls:
    xml_path = rd_path + xml
    image = xml.split('.')[0]
    path_val, label_val, type_val, distribution_val = read_xml(xml_path)
    filtering(image, path_val, label_val, type_val, distribution_val)