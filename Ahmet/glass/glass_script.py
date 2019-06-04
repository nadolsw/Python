# -*- coding: utf-8 -*-
"""
Created on Mon Jun 03 10:29:26 2019

@author: nadolsw

Created based on the following tutorial: http://ataspinar.com/2017/05/26/classification-with-scikit-learn/
"""

#IMPORT NECESSARY PACKAGES
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier

#%% IMPORT & EXPLORE DATA

filename_glass = r"C:\Users\nadolsw\Desktop\Tech\Data Science\Python\Ahmet\glass\glass.csv"
df_glass = pd.read_csv(filename_glass)

print(df_glass.shape)
df_glass.head()
df_glass.describe()
df_glass.isnull().sum() #number of missing values
df_glass.isnull().sum()/len(df_glass)*100 #pct missing values

#%% CORRELATION HEATMAP

correlation_matrix = df_glass.corr()
plt.figure(figsize=(10,8))
ax = sns.heatmap(correlation_matrix, vmax=1, square=True, annot=True,fmt='.2f', cmap ='GnBu', cbar_kws={"shrink": .5}, robust=True)
plt.title('Correlation matrix between the features', fontsize=20)
plt.show()

#%% ABS CORRELATION BAR CHART

def display_corr_with_col(df, col):
    correlation_matrix = df.corr()
    correlation_type = correlation_matrix[col].copy()
    abs_correlation_type = correlation_type.apply(lambda x: abs(x))
    desc_corr_values = abs_correlation_type.sort_values(ascending=False)
    y_values = list(desc_corr_values.values)[1:]
    x_values = range(0,len(y_values))
    xlabels = list(desc_corr_values.keys())[1:]
    fig, ax = plt.subplots(figsize=(8,8))
    ax.bar(x_values, y_values)
    ax.set_title('The correlation of all features with {}'.format(col), fontsize=20)
    ax.set_ylabel('Pearson correlatie coefficient [abs waarde]', fontsize=16)
    plt.xticks(x_values, xlabels, rotation='vertical')
    plt.show()

display_corr_with_col(df_glass, 'Mg')

#%% PCA SCREE PLOT

y_col_glass = 'Type'
x_cols_glass = list(df_glass.columns.values)
x_cols_glass.remove(y_col_glass)

X = df_glass[x_cols_glass].values
X_std = StandardScaler().fit_transform(X)

pca = PCA().fit(X_std)
var_ratio = pca.explained_variance_ratio_
components = pca.components_
#print(pca.explained_variance_)
plt.plot(np.cumsum(var_ratio))
plt.xlim(0,9,1)
plt.xlabel('Number of Features', fontsize=16)
plt.ylabel('Cumulative explained variance', fontsize=16)
plt.show()

#%% PAIRWISE SCATTER PLOTS

ax = sns.pairplot(df_glass, hue='Type')
plt.title('Pairwise relationships between the features')
plt.show()

#%% SPLIT INTO TRAINING & TEST DATASETS

train_test_ratio = 0.7

def get_train_test(df, y_col, x_cols, ratio):

    mask = np.random.rand(len(df)) < train_test_ratio
    df_train = df[mask]
    df_test = df[~mask]
       
    X_train = df_train[x_cols].values
    X_test = df_test[x_cols].values

    Y_train = df_train[y_col].values
    Y_test = df_test[y_col].values

    return df_train, df_test, X_train, Y_train, X_test, Y_test

df_train, df_test, X_train, Y_train, X_test, Y_test = get_train_test(df_glass, y_col_glass, x_cols_glass, train_test_ratio)

#%% CREATE DICTIONARY OF VARIOUS CLASSIFIERS TO TRY

dict_classifiers = {
    "Logistic Regression": LogisticRegression(),
    "Nearest Neighbors": KNeighborsClassifier(),
    "Linear SVM": SVC(),
    "Gradient Boosting Classifier": GradientBoostingClassifier(n_estimators=1000),
    "Decision Tree": tree.DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=1000),
    "Neural Net": MLPClassifier(alpha = 1),
    "Naive Bayes": GaussianNB(),
    "AdaBoost": AdaBoostClassifier(),
    "QDA": QuadraticDiscriminantAnalysis(),
    "Gaussian Process": GaussianProcessClassifier()
}

#%% BATCH CLASSIFIER

def batch_classify(X_train, Y_train, X_test, Y_test, no_classifiers = 11, verbose = True):
    """
    This method, takes as input the X, Y matrices of the Train and Test set.
    And fits them on all of the Classifiers specified in the dict_classifier.
    The trained models, and accuracies are saved in a dictionary. The reason to use a dictionary
    is because it is very easy to save the whole dictionary with the pickle module.
    """
    
    dict_models = {}
    for classifier_name, classifier in list(dict_classifiers.items())[:no_classifiers]:
        t_start = time.clock()
        classifier.fit(X_train, Y_train)
        t_end = time.clock()
        
        t_diff = t_end - t_start
        train_score = classifier.score(X_train, Y_train)
        test_score = classifier.score(X_test, Y_test)
        
        dict_models[classifier_name] = {'model': classifier, 'train_score': train_score, 'test_score': test_score, 'train_time': t_diff}
        if verbose:
            print("trained {c} in {f:.2f} s".format(c=classifier_name, f=t_diff))
    return dict_models

#%% DISPLAY RESULTS

def display_dict_models(dict_models, sort_by='test_score'):
    cls = [key for key in dict_models.keys()]
    test_s = [dict_models[key]['test_score'] for key in cls]
    training_s = [dict_models[key]['train_score'] for key in cls]
    training_t = [dict_models[key]['train_time'] for key in cls]
    
    df_ = pd.DataFrame(data=np.zeros(shape=(len(cls),4)), columns = ['classifier', 'train_score', 'test_score', 'train_time'])
    for ii in range(0,len(cls)):
        df_.loc[ii, 'classifier'] = cls[ii]
        df_.loc[ii, 'train_score'] = training_s[ii]
        df_.loc[ii, 'test_score'] = test_s[ii]
        df_.loc[ii, 'train_time'] = training_t[ii]
        
    print(df_.sort_values(by=sort_by, ascending=False))

#%% EXECUTE PREVIOUS TWO FUNCTIONS TO DETERMINE WHICH CLASSIFIER TO PROCEED WITH
    
dict_models = batch_classify(X_train, Y_train, X_test, Y_test, no_classifiers = 11)
display_dict_models(dict_models)
#GRADIENT BOOSTING WINS

#%% NOW TO OPTIMIZE GB HYPER-PARAMETERS

GDB_params = {
    'n_estimators': [50, 100, 500, 1000, 2000],
    'learning_rate': [0.75, 0.5, 0.1, 0.01, 0.001, 0.0001],
    'criterion': ['friedman_mse', 'mse', 'mae']
}

df_train, df_test, X_train, Y_train, X_test, Y_test = get_train_test(df_glass, y_col_glass, x_cols_glass, 0.6)

for n_est in GDB_params['n_estimators']:
    for lr in GDB_params['learning_rate']:
        for crit in GDB_params['criterion']:
            clf = GradientBoostingClassifier(n_estimators=n_est, 
                                             learning_rate = lr,
                                             criterion = crit)
            clf.fit(X_train, Y_train)
            train_score = clf.score(X_train, Y_train)
            test_score = clf.score(X_test, Y_test)
            print("For ({}, {}, {}) - train, test score: \t {:.5f} \t-\t {:.5f}".format(n_est, lr, crit[:4], train_score, test_score))
            




