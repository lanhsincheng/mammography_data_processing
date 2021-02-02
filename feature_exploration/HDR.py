import numpy as np
import cv2

img_rd_path = r'D:\Mammograph\ROI_training_dataset\maskoverlap_trans/Case_597_CC.jpg'
diff_exposure_list = []

def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def capture_image(img):
    """
            generate multiple images with different exposures.
    """
    diff_exposure_list.append(img)


original = cv2.imread(img_rd_path,0)
# loop over various values of gamma to generate multiple exposure images
for gamma in np.arange(0.0, 1.0, 0.2):
    # ignore when gamma is 1 (there will be no change to the image)
    if gamma == 1:
        # capture_image(original)
        continue
    # apply gamma correction and show the images
    gamma = gamma if gamma > 0 else 0.1
    adjusted = adjust_gamma(original, gamma=gamma)
    capture_image(adjusted)
    # cv2.putText(adjusted, "g={}".format(gamma), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.imshow("Images", np.hstack([original, adjusted]))
    cv2.waitKey(0)

row, col = original.shape
k = 0
f = 0
tt = original.copy()
for i in range(row):
    for j in range(col):
        if diff_exposure_list[4][i][j] - diff_exposure_list[0][i][j]>150:
            tt[i][j] = original[i][j]/2
            k += 1
        else:
            tt[i][j] = original[i][j] + 100
            f += 1
new_arr = ((original - original.min()) * (1/(original.max() - original.min()) * 255)).astype('uint8')
cv2.imshow('new', new_arr)
cv2.waitKey(0)
print(k)
print(f)
# # Recover the Camera Response Function
# # Obtain Camera Response Function (CRF)
# times = np.array([0.5, 1], dtype=np.float32)
# calibrateDebevec = cv2.createCalibrateDebevec()
# responseDebevec = calibrateDebevec.process(diff_exposure_list, times)
# # Merge images into an HDR linear image
# mergeDebevec = cv2.createMergeDebevec()
# hdrDebevec = mergeDebevec.process(diff_exposure_list, times, responseDebevec)
# # Save HDR image.
# cv2.imwrite("hdrDebevec.hdr", hdrDebevec)

# Tonemap using Durand's method obtain 24-bit color image
tonemapDurand = cv2.createTonemapReinhard(1.5,0,0,0)
ldrDurand = tonemapDurand.process(new)
ldrDurand = 3 * ldrDurand
cv2.imshow('new', ldrDurand)
cv2.waitKey(0)
# cv2.imwrite("ldr-Durand.jpg", ldrDurand * 255)

