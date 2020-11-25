"""
    generate auc score and roc curve for modle evaluation
    Usage : generate auc score and roc curve for modle evaluation
    Args:
        path (str): one csv file contain benign and malignant confidence score
"""
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import csv

# read_confidence_score_csv_path
# rd_path1 = r'D:\PycharmProjects\detectron2\projects\mammography_project/integrate_mammo0414_cls_faster_rcnn_R50_fpn_I.csv'
# rd_path2 = r'D:\PycharmProjects\detectron2\projects\mammography_project/mammo0708_ben_cls_faster_rcnn_R50_fpn_mammo0708_aug_contrast.csv'
rd_path1 = r'G:\project_yolo\project_mammo\results/final_csdarknet-omega.csv'
# rd_path2 = r'D:\Mammograph\0_predict_result\yolov4\final\final_gen mal.csv'
# rd_path3 = r'D:\Mammograph\golden\mammo0727_gabor_f0.1_1\final\final_gabor_f01t1.csv'
rd_list = [rd_path1]
# custom split point for naming
split_point = "final_" # split here will get name as "original"
# label answer (1:mal 0:ben)
#y_true = np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1])
# yun
# y_true = np.array([0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0])

#101
y_true =np.array([0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,0,0,1,1,0,0,0,0,0,0,1,1,0,0,1,0,0,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0])

#CBIS-DDSM
#y_true = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,1,1,0,0,0,0,0,0,1,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,1,1,1,1,0,0,1,1,0,0,1,1,1,1,1,1,1,0,0,0,0,1,1,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,1,1,1,0,0,0,0,1,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,1,0,0,1,1,1,1,1,0,0,0,0,0,0,1,1,0,0,0,0,1,1,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,1,1,0,0,1,1,0,0,1,1,0,1,1,0,0,1,1,1,1,0,0,0,0,1,0,0,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,0,0,0,0,1,1,0,1,1,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1])

#CBIS-DDSM for classification
a = [0]*197
b = [1]*129
a.extend(b)
#y_true = np.array(a)
# DDSM
# y_true = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,1,1,0,0,0,0,0,0,1,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,1,1,0,0,1,1,1,1,1,1,1,0,0,0,0,1,1,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,1,1,1,0,0,0,0,1,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,1,0,0,1,1,1,1,1,0,0,0,0,0,0,1,1,0,0,0,0,1,1,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,1,0,0,1,0,0,1,1,0,1,1,0,0,1,1,1,0,0,0,0,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,0,0,0,0,1,1,0,1,1,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1
# ])
# y_scores = np.array([0.763922155, 0.878614664, 0.614958405, 0.703798354, 0.619256198, 0.936455071, 0.999889135, 0, 0.849731386,
#                      0.766337514, 0.873393297, 0, 0.428644538, 0.735836983, 0.454477131, 0, 0.586238563, 0.805306256, 0.899363697,
#                      0, 0.871219933, 0.346833676, 0.384470075, 0.734463632, 0.211915508, 0.185426742, 0.599577725, 0.956777155, 0,
#                      0.36530149, 0, 0.447458684, 0.583866715, 0.198615044, 0.309210598, 0.80309552, 0.224718571, 0, 0.232182726,
#                      0.118217714, 0.281698942, 0.555818498, 0.47232306, 0.285665393, 0.804647267
#                     ])

def read_csv(rd_path):
    benign_list = []
    malignant_list = []
    with open(rd_path, newline='') as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            benign_list.append(float(row[0]))
            malignant_list.append(float(row[1]))
    return benign_list, malignant_list

def acc(name, benign_list, malignant_list):
    ans_list = []
    TP = 0
    for ben_prob, mal_prob in zip(benign_list, malignant_list):
        ans_list.append(0) if ben_prob > mal_prob else ans_list.append(1)
    total = len(ans_list)
    for golden, ans in zip(y_true, ans_list):
        if golden==ans:
            TP += 1
    acc = TP/total
    print(name, " acc is", acc)

def confidence_to_prob(benign_list, malignant_list):
    # output malignant prob
    softmax_output_list = []
    for benign_score, malignant_score in zip(benign_list, malignant_list):
        # detection model
        #inputs = np.array([benign_score, malignant_score])
        #softmax_output = np.exp(inputs)/sum(np.exp(inputs))

        # cls model
        softmax_output = np.array([benign_score, malignant_score])
        softmax_output_list.append(softmax_output)
    return softmax_output_list

def cal_auc_score(name, y_scores):
    print( name, " AUC is", roc_auc_score(y_true, y_scores))

def plot_roc_curve(name, y_scores):

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # fig = plt.figure()
    # lw = 2
    # plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot(fpr, tpr, lw=1, label= name + ' (area = %0.2f)' % roc_auc)
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic example')
    # plt.legend(loc="lower right")
    # # fig.savefig('/tmp/roc.png')
    # plt.show()

# draw baseline figure
fig = plt.figure()
lw = 1
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
for rd_path in rd_list:
    name = rd_path.split(split_point)[1].split(".")[0]
    y_scores = []
    benign_list, malignant_list = read_csv(rd_path)
    acc(name, benign_list, malignant_list)
    softmax_output_list = confidence_to_prob(benign_list, malignant_list)
    for ele in softmax_output_list:
        y_scores.append(ele[1])
    cal_auc_score(name, y_scores)
    plot_roc_curve(name, y_scores)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right")
#fig.savefig('/tmp/roc.png')
plt.show()

