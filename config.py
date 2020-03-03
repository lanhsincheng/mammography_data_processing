"""
    basic several settings, including PATH
    Args:
        path (str): directory, e.g., "D:\Mammograph\original_data\all_labeled_image"
                    elements inside e.g, "D:\Mammograph\original_data\all_labeled_image\Case_1_CC.jpg"
"""

## where to read images
##labeled_jpg path, contain *.jpg
LABEL_IMAGE_PATH = r'D:\Mammograph\original_data\all_label_image_test/'
# LABEL_IMAGE_PATH = r'D:\Mammograph\original_data\0_test/'

##dcm_to_jpg(high_resolution) path, contain *.jpg
DCM_TO_JPG_FOLDER_IMAGE = r'D:\Mammograph\original_data\dcm_to_jpg_folder_image/'

## where to save
##debug_image path, contain *.jpg/png
DEBUG_JPG_PATH = r'D:\Mammograph\final_jpg_debug/'
# DEBUG_JPG_PATH = r'D:\Mammograph\original_data\0_debug_test/'

##cross_match_image path, contain *.jpg/png
CROSS_MATCH_PATH = r'D:\Mammograph\cross_matching/'

##for training data .jpg and .xml
JPEGImages_PATH = r'D:\Mammograph\training_dataset\JPEGImages/'
# JPEGImages_PATH = r'D:\Mammograph\original_data\0_dcm/'
Annotations_PATH = r'D:\Mammograph\training_dataset\Annotations/'

##crop_image path, contain *.jpg/png
CROP_ROI_PATH = r'D:\Mammograph\ROI_training_dataset\JPEGImages/'
# CROP_ROI_PATH = r'D:\Mammograph\original_data\0_crop/'
ROI_Annotation_PATH = r'D:\Mammograph\ROI_training_dataset\Annotations/'