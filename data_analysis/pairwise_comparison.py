import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import random

csv_rd_path = r'G:\project_yolo\000_IN_HOUSE_ANA/Training_Data_Record-value.xlsx'
# csv_rd_path = r'G:\DDSM\CBIS-DDSM/Training_Data_Record.xlsx'
# csv_rd_path = r'G:\project_yolo/Testing_Data_Record.xlsx'

def gen_gaussian_type(df): # only random
    df2 = pd.DataFrame(columns=['case','pathology','type','assessment','subtlety','breast density','calc distribution'], dtype=np.float64)
    # df2 = pd.DataFrame(
    #     columns=['case', 'pathology', 'type', 'assessment', 'subtlety', 'breast density', 'calc distribution', 'benign',
    #              'malignant'], dtype=np.float64)
    # TYPE
    for i in range(len(df['type'])):
        type = df['type'][i]
        if type == 'AMORPHOUS':
            type_val = random.uniform(13,16)
        elif type == 'AMORPHOUS-PLEOMORPHIC':
            type_val = random.uniform(17, 19)
        elif type == 'AMORPHOUS-ROUND_AND_REGULAR':
            type_val = random.uniform(10, 12)
        elif type == 'COARSE':
            type_val = random.uniform(13, 16)
        elif type == 'COARSE-PLEOMORPHIC':
            type_val = random.uniform(17, 19)
        elif type == 'COARSE-ROUND_AND_REGULAR':
            type_val = random.uniform(13, 16)
        elif type == 'COARSE-ROUND_AND_REGULAR-LUCENT_CENTER':
            type_val = random.uniform(10, 12)
        elif type == 'COARSE-ROUND_AND_REGULAR-LUCENT_CENTERED':
            type_val = random.uniform(10, 12)
        elif type == 'DYSTROPHIC':
            type_val = random.uniform(0, 2)
        elif type == 'EGGSHELL':
            type_val = random.uniform(0, 2)
        elif type == 'FINE_LINEAR_BRANCHING':
            type_val = random.uniform(27, 29)
        elif type == 'LARGE_RODLIKE':
            type_val = random.uniform(0, 2)
        elif type == 'LARGE_RODLIKE-ROUND_AND_REGULAR':
            type_val = random.uniform(0, 2)
        elif type == 'LUCENT_CENTER':
            type_val = random.uniform(0, 2)
        elif type == 'LUCENT_CENTERED':
            type_val = random.uniform(0, 2, )
        elif type == 'LUCENT_CENTER-PUNCTATE':
            type_val = random.uniform(7, 9)
        elif type == 'MILK_OF_CALCIUM':
            type_val = random.uniform(0, 2)
        elif type == 'N/A' or 'nan':
            type_val = random.uniform(0, 20)
        elif type == 'PLEOMORPHIC':
            type_val = random.uniform(27, 29)
        elif type == 'PLEOMORPHIC-FINE_LINEAR_BRANCHING':
            type_val = random.uniform(27, 29)
        elif type == 'PLEOMORPHIC-PLEOMORPHIC':
            type_val = random.uniform(27, 29)
        elif type == 'PUNCTATE':
            type_val = random.uniform(13, 16)
        elif type == 'PUNCTATE-AMORPHOUS':
            type_val = random.uniform(13, 16)
        elif type == 'PUNCTATE-FINE_LINEAR_BRANCHING':
            type_val = random.uniform(17, 19)
        elif type == 'PUNCTATE-LUCENT_CENTER':
            type_val = random.uniform(10, 12)
        elif type == 'PUNCTATE-PLEOMORPHIC':
            type_val = random.uniform(17, 19)
        elif type == 'PUNCTATE-ROUND_AND_REGULAR':
            type_val = random.uniform(10, 12)
        elif type == 'ROUND_AND_REGULAR':
            type_val = random.uniform(0, 2)
        elif type == 'ROUND_AND_REGULAR-AMORPHOUS':
            type_val = random.uniform(7, 9)
        elif type == 'ROUND_AND_REGULAR-EGGSHELL':
            type_val = random.uniform(0, 2)
        elif type == 'ROUND_AND_REGULAR-LUCENT_CENTER':
            type_val = random.uniform(0, 2)
        elif type == 'ROUND_AND_REGULAR-LUCENT_CENTER-DYSTROPHIC':
            type_val = random.uniform(0, 2)
        elif type == 'ROUND_AND_REGULAR-LUCENT_CENTERED':
            type_val = random.uniform(0, 2)
        elif type == 'ROUND_AND_REGULAR-LUCENT_CENTER-PUNCTATE':
            type_val = random.uniform(7, 9)
        elif type == 'ROUND_AND_REGULAR-PLEOMORPHIC':
            type_val = random.uniform(7, 9)
        elif type == 'ROUND_AND_REGULAR-PUNCTATE':
            type_val = random.uniform(7, 9)
        elif type == 'ROUND_AND_REGULAR-PUNCTATE-AMORPHOUS':
            type_val = random.uniform(7, 9)
        elif type == 'SKIN':
            type_val = random.uniform(0, 2)
        elif type == 'SKIN-COARSE-ROUND_AND_REGULAR':
            type_val = random.uniform(7, 9)
        elif type == 'SKIN-PUNCTATE':
            type_val = random.uniform(3, 6)
        elif type == 'SKIN-PUNCTATE-ROUND_AND_REGULAR':
            type_val = random.uniform(0, 2)
        elif type == 'VASCULAR':
            type_val = random.uniform(0, 2)
        elif type == 'VASCULAR-COARSE':
            type_val = random.uniform(7, 9)
        elif type == 'VASCULAR-COARSE-LUCENT_CENTERED':
            type_val = random.uniform(0, 2)
        elif type == 'VASCULAR-COARSE-LUCENT_CENTER-ROUND_AND_REGULAR-PUNCTATE':
            type_val = random.uniform(0, 2)
        df2.loc[i,'type'] = round(type_val,5)
    # DISTRIBUTION
    for i in range(len(df['calc distribution'])):
        dist = df['calc distribution'][i]
        if dist == 'CLUSTERED':
            dist_val = random.uniform(10,19)
        elif dist == 'CLUSTERED-LINEAR':
            dist_val = random.uniform(10, 19)
        elif dist == 'CLUSTERED-SEGMENTAL':
            dist_val = random.uniform(10, 19)
        elif dist == 'DIFFUSELY_SCATTERED':
            dist_val = random.uniform(0, 9) #0-9
        elif dist == 'LINEAR':
            dist_val = random.uniform(20, 29)
        elif dist == 'LINEAR-SEGMENTAL':
            dist_val = random.uniform(20, 29)
        elif dist == 'N/A'or 'nan':
            dist_val = random.uniform(0, 9)
        elif dist == 'REGIONAL':
            dist_val = random.uniform(10, 19) #10-19
        elif dist == 'REGIONAL-REGIONAL':
            dist_val = random.uniform(10, 19) #10-19
        elif dist == 'SEGMENTAL':
            dist_val = random.uniform(20, 29)
        df2.loc[i, 'calc distribution'] = round(dist_val, 5)

    df2['case'] = df['case']
    df2['pathology'] = df['pathology']
    df2['assessment'] = df['assessment']
    df2['subtlety'] = df['subtlety']
    df2['breast density'] = df['breast density']
    # df2['calc distribution'] = df['calc distribution']

    # df2['benign'] = df['benign']
    # df2['malignant'] = df['malignant']
    # print(df2.head())

    return df2

def IN_BREAST_gen_gaussian_type(df): # only random
    df2 = pd.DataFrame(columns=['case','pathology','type','age','calc distribution'], dtype=np.float64)

    # TYPE
    for i in range(len(df['type'])):
        type = df['type'][i]
        if type == 'Amorphous':
            type_val = random.uniform(20,30)
        elif type == 'Coarse heterogeneous':
            type_val = random.uniform(20, 30)
        elif type == 'Fine linear/linear branching' or type =='Fine linear':
            type_val = random.uniform(40, 50)
        elif type == 'Fine pleomorphic' or type =='Pleomorphic':
            type_val = random.uniform(30, 40)
        elif type == 'Linear':
            type_val = random.uniform(40, 50)
        elif type == 'Linear branching':
            type_val = random.uniform(40, 50)
        elif type == 'Pleomorphic, Coarse heterogeneous':
            type_val = random.uniform(30, 40)
        elif type == 'Punctate':
            type_val = random.uniform(10, 20)
        elif type == 'Round':
            type_val = random.uniform(10, 20)

        df2.loc[i,'type'] = round(type_val,5)
    # DISTRIBUTION
    for i in range(len(df['calc distribution'])):
        dist = df['calc distribution'][i]
        if dist == 'Group' or dist == 'Grouped':
            dist_val = random.uniform(10,30)
        elif dist == 'Grouped/regional':
            dist_val = random.uniform(10, 30)
        elif dist == 'Linear':
            dist_val = random.uniform(30, 50)
        elif dist == 'Regional':
            dist_val = random.uniform(10, 30)
        elif dist == 'Segmental':
            dist_val = random.uniform(30, 50)

        df2.loc[i, 'calc distribution'] = round(dist_val, 5)

    df2['case'] = df['case']
    df2['pathology'] = df['pathology']
    df2['age'] = df['age']

    # df2['benign'] = df['benign']
    # df2['malignant'] = df['malignant']
    # print(df2.head())

    return df2


df = pd.read_excel (csv_rd_path)
# print (df)

# df2 = gen_gaussian_type(df) # ddsm
df2 = IN_BREAST_gen_gaussian_type(df)

# print(df2['log_type'].head())
sns.pairplot(df2,vars=['type','calc distribution','age'],hue="pathology")
plt.show()