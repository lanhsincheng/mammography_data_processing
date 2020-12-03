"""
        Convert INBreast official data(segmentation) xml to voc format xml
        Usage : change path
        Args:
            data csv_file (str): e.g., "D:\Mammograph\original_data\all_labeled_image.csv"
            data img_folder (str): e.g., "D:\Mammograph\original_data\all_labeled_image"
        Returns:
            each voc format xml file
    """
import os
import plistlib
import cv2
import xml.etree.cElementTree as ET

# inbreast xml path, contain *.xml
xml_path = r'D:\Mammograph\reference_dataset\INbreast Release 1.0\INbreast Release 1.0\AllXML/'
# inbreast img path, contain *.jpg
img_path_root = r'D:\Mammograph\reference_dataset\INbreast Release 1.0\INbreast Release 1.0\INbreast_DCM_JPG/'
xml_wb_root =r'D:\Mammograph\reference_dataset\INbreast Release 1.0\INbreast Release 1.0\voc_format_xml/'

all_label_dirs = os.listdir(xml_path)


def form_voc_format_xml(img_path, twidth, theight, tdepth, txmin, tymin, txmax, tymax, img_name, class_name):
    """
        This function form xml file in voc format
        Args:
            path, width, height, depth, xmin, ymin, xmax, ymax, img_name, class_name info for xml
        Returns:
            no returns
            write back to xml file
    """
    if os.path.isfile(xml_wb_root + img_name + '.xml') :
        tree = ET.parse(xml_wb_root + img_name + '.xml')
        xmlRoot = tree.getroot()
        object = ET.Element('object')
        xmlRoot.append(object)
        name = ET.SubElement(object, 'name')
        name.text = class_name
        pose = ET.SubElement(object, 'pose')
        pose.text = 'Unspecified'
        truncated = ET.SubElement(object, 'truncated')
        truncated.text = '0'
        difficult = ET.SubElement(object, 'difficult')
        difficult.text = '0'
        bndbox = ET.SubElement(object, 'bndbox')
        xmin = ET.SubElement(bndbox, 'xmin')
        xmin.text = str(txmin)
        ymin = ET.SubElement(bndbox, 'ymin')
        ymin.text = str(tymin)
        xmax = ET.SubElement(bndbox, 'xmax')
        xmax.text = str(txmax)
        ymax = ET.SubElement(bndbox, 'ymax')
        ymax.text = str(tymax)

        wb_xml_path = xml_wb_root + img_name + '.xml'
        tree.write(wb_xml_path)

    else:
        annotation = ET.Element('annotation')
        folder = ET.SubElement(annotation, 'folder')
        folder.text = '0'
        filename = ET.SubElement(annotation, 'filename')
        filename.text = img_name + '.jpg'
        path = ET.SubElement(annotation, 'path')
        path.text = img_path
        source = ET.SubElement(annotation, 'source')
        database = ET.SubElement(source, 'database')
        database.text = 'Unknown'
        size = ET.SubElement(annotation, 'size')
        width = ET.SubElement(size, 'width')
        width.text = str(twidth)
        height = ET.SubElement(size, 'height')
        height.text = str(theight)
        depth = ET.SubElement(size, 'depth')
        depth.text = str(tdepth)
        segmented = ET.SubElement(annotation, 'segmented')
        segmented.text = '0'
        object = ET.SubElement(annotation, 'object')
        name = ET.SubElement(object, 'name')
        name.text = class_name
        pose = ET.SubElement(object, 'pose')
        pose.text = 'Unspecified'
        truncated = ET.SubElement(object, 'truncated')
        truncated.text = '0'
        difficult = ET.SubElement(object, 'difficult')
        difficult.text = '0'
        bndbox = ET.SubElement(object, 'bndbox')
        xmin = ET.SubElement(bndbox, 'xmin')
        xmin.text = str(txmin)
        ymin = ET.SubElement(bndbox, 'ymin')
        ymin.text = str(tymin)
        xmax = ET.SubElement(bndbox, 'xmax')
        xmax.text = str(txmax)
        ymax = ET.SubElement(bndbox, 'ymax')
        ymax.text = str(tymax)

        wb_xml_path = xml_wb_root + img_name + '.xml'
        myfile = open(wb_xml_path, "w")
        myfile.write(ET.tostring(annotation).decode("utf-8"))

def load_img_basic_info(img_name):
    """
        This function reads image to get its basic info  e.g., filename, path, width, height, depth
        Args:
            img_name(str) : img_name which are need to be coverted
        Returns:
            str : path
            int : width, height, depth
    """
    img_path = os.path.join(img_path_root, img_name + '.jpg')
    img = cv2.imread(img_path)
    height, width, depth =img.shape
    return img_path, width, height, depth

def load_inbreast(xml_path):
    """
        This function loads a osirix xml region as a binary numpy array for INBREAST dataset
        Args:
            xml_path(str) : path to xml which are need to be coverted
        Returns:
            int : xmin, ymin, xmax, ymax
            str : name(class name)
    """

    def load_point(point_string):
        x, y = tuple([float(num) for num in point_string.strip('()').split(',')])
        return x, y

    with open(xml_path,'rb') as anno_file:
        img_name = xml_path.split("/")[1].split(".")[0]
        # img = cv2.imread(r'D:\Mammograph\reference_dataset\INbreast Release 1.0\INbreast Release 1.0\test/20586908.jpg')
        plist_dict = plistlib.load(anno_file, fmt=plistlib.FMT_XML)['Images'][0]
        numRois = plist_dict['NumberOfROIs']
        print(numRois)
        rois = plist_dict['ROIs']
        assert len(rois) == numRois
        print(len(rois))
        for roi in rois:
            numPoints = roi['NumberOfPoints']
            points = roi['Point_px']
            assert numPoints == len(points)
            class_name = roi['Name'] #Calcification
            points = [load_point(point) for point in points]
            if len(points) == 1:
                xmin = int(points[0][0]-15)
                ymin = int(points[0][1]-15)
                xmax = int(points[0][0]+15)
                ymax = int(points[0][1]+15)
                print(xmin, ymin,xmax, ymax)
                img_path, width, height, depth = load_img_basic_info(img_name)
                # img = cv2.rectangle(img, (xmin, ymin),(xmax, ymax), (0, 255, 0), 3)
                # p =r'D:\Mammograph\reference_dataset\INbreast Release 1.0\INbreast Release 1.0\test/'
                # cv2.imwrite(p+img_name+'.jpg',img)
                form_voc_format_xml(img_path, width, height, depth, xmin, ymin, xmax, ymax, img_name, class_name)
            else:
                xlist = []
                ylist = []
                for point in points:
                    xlist.append(point[0])
                    ylist.append(point[1])
                xmin = int(min(xlist))-1
                ymin = int(min(ylist))-1
                xmax = int(max(xlist))+1
                ymax = int(max(ylist))+1
                print(xmin, ymin,xmax, ymax)
                img_path, width, height, depth = load_img_basic_info(img_name)
                # img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
                # p = r'D:\Mammograph\reference_dataset\INbreast Release 1.0\INbreast Release 1.0\test/'
                # cv2.imwrite(p + img_name + '.jpg', img)
                form_voc_format_xml(img_path, width, height, depth, xmin, ymin, xmax, ymax, img_name, class_name)

for xmlfile in all_label_dirs:
    loaded_xml_path = xml_path + xmlfile
    load_inbreast(loaded_xml_path)