"""
        Pick up image corresponding to xml
        Usage : change path
        Args:
            data csv_file (str): e.g., "D:\Mammograph\original_data\all_labeled_image.csv"
            data img_folder (str): e.g., "D:\Mammograph\original_data\all_labeled_image"
        Returns:
            pickup images
    """
import os
import cv2
# inbreast xml path, contain *.xml
xml_path = r'D:\Mammograph\reference_dataset\INbreast Release 1.0\INbreast Release 1.0\AllXML'
jpg_path = r'D:\Mammograph\reference_dataset\INbreast Release 1.0\INbreast Release 1.0\INbreast_DCM_JPG/'
all_xml_dirs = os.listdir(xml_path)

for xml_file in all_xml_dirs:
    fetch_jpg_path = os.path.join(jpg_path,  xml_file.split('.')[0] + '.jpg')
    img = cv2.imread(fetch_jpg_path)
    pickup_folder = r'D:\Mammograph\reference_dataset\INbreast Release 1.0\INbreast Release 1.0\JPEGImages/'
    cv2.imwrite(pickup_folder + xml_file.split('.')[0] + '.jpg', img)