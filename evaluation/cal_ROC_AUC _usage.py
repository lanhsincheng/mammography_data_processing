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
rd_path1 = r'D:\Mammograph\golden\mammo0719_original\final\final_original.csv'
rd_path2 = r'D:\Mammograph\final_ans1.csv'
rd_path3 = r'D:\Mammograph\final_ans2.csv'
rd_list = [rd_path1, rd_path2, rd_path3]
# custom split point for naming
split_point = "final_" # split here will get name as "original", "ans1", "ans2" 
# label answer (0 for benign, 1for malignant)
y_true = np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1])

def read_csv(rd_path):
    benign_list = []
    malignant_list = []
    with open(rd_path, newline='') as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            benign_list.append(float(row[0]))
            malignant_list.append(float(row[1]))
    return benign_list, malignant_list

def confidence_to_prob(benign_list, malignant_list):
    # output malignant prob
    softmax_output_list = []
    for benign_score, malignant_score in zip(benign_list, malignant_list):
        inputs = np.array([benign_score, malignant_score])
        softmax_output = np.exp(inputs)/sum(np.exp(inputs))
        softmax_output_list.append(softmax_output)
    return softmax_output_list

def cal_auc_score(name, y_scores):
    print( name, " AUC is", roc_auc_score(y_true, y_scores))

def plot_roc_curve(name, y_scores):

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=1, label= name + ' (area = %0.2f)' % roc_auc)

# draw baseline figure
fig = plt.figure()
lw = 1
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
for rd_path in rd_list:
    name = rd_path.split(split_point)[1].split(".")[0]
    y_scores = []
    benign_list, malignant_list = read_csv(rd_path)
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

