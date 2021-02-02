import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import random
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# train_csv_rd_path_inhouse = r'G:\project_yolo\000_IN_HOUSE_ANA\32_1121_all_original_V2/Training_Data_Record-32.xlsx'
train_csv_rd_path_inhouse = r'G:\project_yolo\000_IN_HOUSE_ANA\47/Training_Data_Record-47_4000train.xlsx'
# train_csv_rd_path_inhouse = r'G:\project_yolo\000_IN_HOUSE_ANA\32_1121_all_original_V2/Validation_Data_Record.xlsx'
test_csv_rd_path_inhouse = r'G:\project_yolo\000_IN_HOUSE_ANA\32_1121_all_original_V2/Testing_Data_Record.xlsx'
# train_csv_rd_path_cbisddsm = r'G:\project_yolo\000_CBIS_DDSM_ANA\18_1105_DDSM_original_ASFF_alphaV2/Training_Data_Record.xlsx'
# train_csv_rd_path_cbisddsm = r'G:\project_yolo\000_CBIS_DDSM_ANA\18_1105_DDSM_original_ASFF_alphaV2/Validation_Training_Data_Record-1546.xlsx'
train_csv_rd_path_cbisddsm = r'G:\project_yolo\000_CBIS_DDSM_ANA\14_1024_DDSM_original_anchor/Validation_Training_Data_Record.xlsx'
test_csv_rd_path_cbisddsm = r'G:\project_yolo\000_CBIS_DDSM_ANA\18_1105_DDSM_original_ASFF_alphaV2/Validation_Testing_Data_Record.xlsx'
testset_size = 170
#
# parameter_space = {
#     'hidden_layer_sizes': [(10,10,20,20,50,50,50),(50,50,50,100,100,50,50),(10,10,50,50,50,80,80,80,80,100),(50,50,100,100,50,50)],
#     'activation': ['identity', 'tanh', 'relu','logistic'],
#     'solver': ['adam','lbfgs','sgd'],
#     'alpha': [0.0001, 0.05],
#     'learning_rate': ['constant','adaptive'],
#     'learning_rate_init': [0.0001, 0.001, 0.005],
#     'random_state': [0, 1],
#     'validation_fraction': [0.1, 0.15, 0.2]
# }

# parameter_space = {
#     'hidden_layer_sizes': [(5,5)],
#     # 'hidden_layer_sizes': [(50,100,100,50)],
#     'activation': ['tanh'],
#     'solver': ['adam'],
#     'alpha': [0.08], #0.08
#     'learning_rate': ['adaptive'],
#     'learning_rate_init': [0.005], #0.005
#     # 'random_state': [0, 1],
#     'validation_fraction': [ 0.2]
# }

## best of in-house original
parameter_space = {
    'hidden_layer_sizes': [(5,5)],
    # 'hidden_layer_sizes': [(50,100,100,50)],
    'activation': ['relu'], #32:identity
    'solver': ['adam'],
    'alpha': [0.0001], #32:0.0001
    'learning_rate': ['adaptive'],
    'learning_rate_init': [0.005], #0.005
    # 'random_state': [0, 1],
    'validation_fraction': [0.2]
}

## best of in-house mask
# parameter_space = {
#     'hidden_layer_sizes': [(5,5),(10,10,20,20,50,50,50),(50,50,100,100,50,50)],
#     # 'hidden_layer_sizes': [(50,100,100,50)],
#     'activation': ['identity', 'tanh', 'relu','logistic'],
#     'solver': ['adam','sgd'],
#     'alpha': [ 0.0001, 0.05],
#     'learning_rate': ['constant','adaptive'],
#     'learning_rate_init': [0.0001, 0.001, 0.005], #0.005
#     # 'random_state': [0, 1],
#     'validation_fraction': [0.1, 0.15, 0.2]
# }

# parameter_space = {
#     # 'hidden_layer_sizes': [(5,5)],
#     'hidden_layer_sizes': [(10,10,20,20,50,50,50)],
#     'activation': ['identity'],
#     'solver': ['adam'],
#     'alpha': [ 0.05],
#     'learning_rate': ['constant'],
#     'learning_rate_init': [0.05], #0.05
#     # 'random_state': [0, 1],
#     'validation_fraction': [0.15]
# }

def make_classification_set(train_label, train_feature, train_type, train_distribution, train_benign, train_malignant, train_height, train_width, train_area, train_ratio):
    X_train = []
    X_test = []

    for i1, i2, i3, i4, i5, i6, i7, i8, i9 in zip(train_feature, train_type, train_distribution, train_benign, train_malignant, train_height, train_width, train_area, train_ratio):
        X_train.append([i1, i2, i3, i4, i5, i6, i7, i8, i9])
    # for i1, i2, i3, i4, i5, i6, i7, i8, i9 in zip(test_feature, test_type, test_distribution, test_benign, test_malignant, test_height, test_width, test_area, test_ratio):
    #     X_test.append([i1, i2, i3, i4, i5, i6, i7, i8, i9])

    # return np.array(X_train), np.array(X_test), np.array(train_label), np.array(test_label)
    # return np.array(X_train), np.array(train_label)
    return X_train, train_label

def CBIS_DDSM_gen_gaussian_type(df):  # only random

    case = df['case'].tolist()
    patho = df['pathology'].tolist()
    type = df['type'].tolist()
    assessment = df['assessment'].tolist()
    subtlety = df['subtlety'].tolist()
    breastdensity = df['breastdensity'].tolist()
    distribution = df['calcdistribution'].tolist()
    benign = df['benign'].tolist()
    malignant = df['malignant'].tolist()
    height = df['height'].tolist()
    width = df['width'].tolist()
    area = df['area'].tolist()
    ratio = df['ratio(w/h)'].tolist()
    v2benign = df['v2benign'].tolist()
    v2malignant = df['v2malignant'].tolist()

    df_access = pd.DataFrame({'case': case,
                              'pathology': patho,
                              'type': type,
                              'assessment': assessment,
                              'subtlety': subtlety,
                              'breastdensity': breastdensity,
                              'calcdistribution': distribution,
                              'benign': benign,
                              'malignant': malignant,
                              'height': height,
                              'width': width,
                              'area': area,
                              'ratio(w/h)': ratio,
                              'v2benign': v2benign,
                              'v2malignant': v2malignant},
                             columns=['case', 'pathology', 'type', 'assessment','subtlety', 'breastdensity','calcdistribution', 'benign', 'malignant',
                                      'height', 'width', 'area', 'ratio(w/h)', 'v2benign', 'v2malignant'], dtype=np.float64)

    df_not_encoded = df_access.set_index('case')
    data_type = df_access.set_index('case').type.str.split('-', expand=True).stack()
    data_distribution = df_access.set_index('case').calcdistribution.str.split('-', expand=True).stack()
    data_dum_type = pd.get_dummies(data_type).groupby(level=0).sum()
    data_dum_distribution = pd.get_dummies(data_distribution).groupby(level=0).sum()
    # type = data_dum_type.reindex(
    #     index=data_dum_type.index.to_series().str.rsplit('_').str[0].sort_values().index)
    # distribution = data_dum_distribution.reindex(
    #     index=data_dum_distribution.index.to_series().str.rsplit('_').str[0].sort_values().index)
    type_list = data_dum_type.values.tolist()
    distribution_list = data_dum_distribution.values.tolist()

    # concate
    p = pd.concat([data_dum_type, data_dum_distribution], axis=1)
    # p = p.reindex(index=p.index.to_series().str.rsplit('_').str[-2].astype(int).sort_values().index)
    # p.reset_index(drop=True, inplace=True)
    # df_not_encoded.reset_index(drop=True, inplace=True)

    # df_final = df_not_encoded
    df_final = pd.concat([df_not_encoded, p], axis=1)
    df_final.drop(['type', 'calcdistribution'], axis=1, inplace=True)

    sc = StandardScaler()
    df_final[['benign', 'malignant', 'height', 'width', 'area', 'ratio(w/h)','breastdensity','assessment','subtlety','v2benign', 'v2malignant']] = sc.fit_transform(
        df_final[['benign', 'malignant', 'height', 'width', 'area', 'ratio(w/h)','breastdensity','assessment','subtlety','v2benign', 'v2malignant']])

    patho = df_final['pathology'].tolist()
    breastdensity = df_final['breastdensity'].tolist()
    benign = df_final['benign'].tolist()
    malignant = df_final['malignant'].tolist()
    height = df_final['height'].tolist()
    width = df_final['width'].tolist()
    area = df_final['area'].tolist()
    ratio = df_final['ratio(w/h)'].tolist()
    v2benign = df_final['v2benign'].tolist()
    v2malignant = df_final['v2malignant'].tolist()

    # df_final.drop(['benign', 'height', 'width', 'breastdensity', 'assessment', 'subtlety','area', 'ratio(w/h)'], axis=1, inplace=True)
    df_final.drop(['height', 'width', 'assessment', 'subtlety', 'breastdensity','area', 'ratio(w/h)', 'benign', 'malignant','v2benign', 'v2malignant'], axis=1,inplace=True)

    return df_final, patho, breastdensity, np.array(type_list, dtype=np.float32), np.array(distribution_list, dtype=np.float32), benign, malignant, height, width, area, ratio



def IN_HOUSE_gen_gaussian_type(df):  # only random

    # df_access = pd.DataFrame(columns=['case', 'pathology', 'type', 'age', 'calc  distribution', 'benign', 'malignant'], dtype=np.float64)
    case = df['case'].tolist()
    patho = df['pathology'].tolist()
    type = df['type'].tolist()
    age = df['age'].tolist()
    distribution = df['calcdistribution'].tolist()
    benign = df['benign'].tolist()
    malignant = df['malignant'].tolist()
    height = df['height'].tolist()
    width = df['width'].tolist()
    area = df['area'].tolist()
    ratio = df['ratio(w/h)'].tolist()
    mask_benign = df['maskbenign'].tolist()
    mask_malignant = df['maskmalignant'].tolist()
    maskans = df['maskans'].tolist()
    originalans = df['originalans'].tolist()

    df_access = pd.DataFrame({'case':case,
                              'pathology': patho,
                              'type': type,
                              'age': age,
                              'calcdistribution': distribution,
                              'benign': benign,
                              'malignant': malignant,
                              'height': height,
                              'width': width,
                              'area': area,
                              'ratio(w/h)': ratio,
                              'maskbenign': mask_benign,
                              'maskmalignant': mask_malignant,
                              'maskans': maskans,
                              'originalans': originalans
                              },
                             columns=['case', 'pathology', 'type', 'age', 'calcdistribution', 'benign', 'malignant', 'height', 'width', 'area', 'ratio(w/h)', 'maskbenign', 'maskmalignant', 'maskans', 'originalans'], dtype=np.float64)

    df_not_encoded = df_access.set_index('case')
    data_type = df_access.set_index('case').type.str.split('|', expand=True).stack()
    data_distribution = df_access.set_index('case').calcdistribution.str.split('|', expand=True).stack()
    data_dum_type = pd.get_dummies(data_type).groupby(level=0).sum()
    data_dum_distribution = pd.get_dummies(data_distribution).groupby(level=0).sum()
    type = data_dum_type.reindex(index=data_dum_type.index.to_series().str.rsplit('_').str[-2].astype(int).sort_values().index)
    distribution = data_dum_distribution.reindex(index=data_dum_distribution.index.to_series().str.rsplit('_').str[-2].astype(int).sort_values().index)
    type_list =type.values.tolist()
    distribution_list = distribution.values.tolist()

    # concate
    p = pd.concat([type, distribution], axis=1)
    # p = p.reindex(index=p.index.to_series().str.rsplit('_').str[-2].astype(int).sort_values().index)
    # p.reset_index(drop=True, inplace=True)
    # df_not_encoded.reset_index(drop=True, inplace=True)

    # df_final = df_not_encoded
    df_final = pd.concat([df_not_encoded, p], axis=1)
    df_final.drop(['type', 'calcdistribution'], axis=1, inplace=True)

    sc = StandardScaler()
    df_final[['age', 'benign', 'malignant', 'height', 'width', 'area', 'ratio(w/h)', 'maskbenign', 'maskmalignant']] = sc.fit_transform(df_final[['age', 'benign', 'malignant', 'height', 'width', 'area', 'ratio(w/h)','maskbenign', 'maskmalignant']])
    # df_final.drop([ 'height', 'width','area', 'ratio(w/h)'], axis=1, inplace=True)


    patho = df_final['pathology'].tolist()
    age = df_final['age'].tolist()
    benign = df_final['benign'].tolist()
    malignant = df_final['malignant'].tolist()
    height = df_final['height'].tolist()
    width = df_final['width'].tolist()
    area = df_final['area'].tolist()
    ratio = df_final['ratio(w/h)'].tolist()

    # df_final.drop(['height', 'width', 'maskans','originalans'], axis=1, inplace=True)
    # df_final.drop(['height', 'width','maskans','originalans','area','ratio(w/h)', 'benign', 'maskbenign'], axis=1, inplace=True)
    # df_final.drop(['maskans', 'originalans','height', 'width', 'maskbenign', 'maskmalignant'], axis=1, inplace=True)
    df_final.drop(['height', 'width','maskans','originalans','area'], axis=1, inplace=True)

    return df_final, patho, age, np.array(type_list,dtype=np.float32), np.array(distribution_list,dtype=np.float32), benign, malignant, height, width, area, ratio

def gen_gaussian_type(df, func):
    df_final, patho, feature, type, distribution, benign, malignant, height, width, area, ratio = func[0](df)

    return df_final, patho, feature, type, distribution, benign, malignant, height, width, area, ratio

def MLP_classifier(X_train, y_train):
    # clf = MLPClassifier(activation='tanh',shuffle = True, warm_start=True, learning_rate_init=0.0001, random_state=1, max_iter=10000,verbose=True)

    mlp_gs = MLPClassifier(max_iter=100000)
    clf = GridSearchCV(mlp_gs, parameter_space, n_jobs=-1, cv=10)
    clf = clf.fit(X_train, y_train)
    print('Best parameters found:\n', clf.best_params_)
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    return clf.best_estimator_

def classifier_func(df, classifier_func, X_train, y_train):
    clf = classifier_func[0](X_train, y_train)

    return clf

def predict_and_write(clf, X_train, X_test, y_train, y_test):
    prob = clf.predict_proba(X_test)
    predict = clf.predict(X_test)
    score = clf.score(X_test, y_test)
    print("Training set score: %f" % clf.score(X_train, y_train))
    print("Test set score: %f" % clf.score(X_test, y_test))
    # print('acc: ', score)

    # write into txt
    with open('mlp.txt', "w") as f:
        f.write("\n".join("\n".join(map(str, x)) for x in (prob, predict)))
    with open('mlp-weights.txt', "w") as f:
        f.write("\n".join("".join(map(str, x)) for x in (clf.coefs_, clf.intercepts_)))
    return score

def run(params):
    dataset = params["dataset"]
    func = params["func"]
    classifier = params["meta_classifier"]
    if dataset == ['In-House']:
        train_path = train_csv_rd_path_inhouse
        test_path = test_csv_rd_path_inhouse
    else:
        train_path = train_csv_rd_path_cbisddsm
        test_path = test_csv_rd_path_cbisddsm
    df_train = pd.read_excel(train_path)
    df_test = pd.read_excel(test_path)
    train_df_final, train_label, train_feature, train_type, train_distribution, train_benign, train_malignant, train_height, train_width, train_area, train_ratio = gen_gaussian_type(df_train, func)
    # test_df_final, test_label, test_feature, test_type, test_distribution, test_benign, test_malignant, test_height, test_width, test_area, test_ratio = gen_gaussian_type(df_test, func)

    ## df split train val
    X_train= train_df_final.values[:-testset_size,1:]
    y_train = train_df_final.values[:-testset_size,0]
    X_test = train_df_final.values[-testset_size:, 1:]
    y_test = train_df_final.values[-testset_size:, 0]
    ## df split train test
    # X_test = train_df_final.values[:-testset_size, 1:]
    # y_test = train_df_final.values[:-testset_size, 0]
    # X_train = train_df_final.values[-testset_size:, 1:]
    # y_train = train_df_final.values[-testset_size:, 0]


    # X_train, X_test, y_train, y_test = make_classification_set(train_label, train_feature, train_type, train_distribution, train_benign, train_malignant, train_height, train_width, train_area, train_ratio,
    #                                                            test_label, test_feature, test_type, test_distribution, test_benign, test_malignant, test_height, test_width, test_area, test_ratio)
    # X,y = make_classification_set(train_label, train_feature, train_type,train_distribution, train_benign, train_malignant,train_height, train_width, train_area, train_ratio)
    # X_train = X[:-101,:]
    # y_train = y[:-101]
    # X_test = X[-101:, :]
    # y_test = y[-101:]

    # clf = classifier_func(df_train, classifier, X_train, y_train)
    # score = predict_and_write(clf, X_train, X_test, y_train, y_test)
    score_list = []
    for i in range(50):
        clf = classifier_func(df_train, classifier, X_train, y_train)
        score = predict_and_write(clf, X_train, X_test, y_train, y_test)
        score_list.append(score)
    mean = sum(score_list) / len(score_list)
    std = (sum([((x - mean) ** 2) for x in score_list]) / len(score_list))** 0.5
    print('mean:',mean, 'std:',std)

params = {
    # "dataset" : ['In-House', 'CBIS-DDSM'],
    # "func" : [IN_HOUSE_gen_gaussian_type, CBIS_DDSM_gen_gaussian_type],
    # "meta_classifier" : [MLP_classifier, MLP_classifier]

    # "dataset" : ['CBIS-DDSM'],
    # "func" : [CBIS_DDSM_gen_gaussian_type],
    # "meta_classifier" : [MLP_classifier]

    "dataset" : ['In-House'],
    "func" : [IN_HOUSE_gen_gaussian_type],
    "meta_classifier" : [MLP_classifier]
}
run(params)

