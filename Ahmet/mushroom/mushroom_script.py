# -*- coding: utf-8 -*-
"""
Created on Mon Jun 03 11:47:27 2019

@author: nadolsw

Created based on the following tutorial: http://ataspinar.com/2017/05/26/classification-with-scikit-learn/
"""

#IMPORT NECESSARY PACKAGES

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
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

#%% IMPORT DATA

filename_mushrooms = r"C:\Users\nadolsw\Desktop\Tech\Data Science\Python\Ahmet\mushroom\mushrooms.csv"
df_mushrooms = pd.read_csv(filename_mushrooms)

#%% PRINT OUT SUBSET AND UNIQUE VALUES

display(df_mushrooms.head())

for col in df_mushrooms.columns.values:
    print(col, df_mushrooms[col].unique())

#Remove vars with only single value
for col in df_mushrooms.columns.values:
    if len(df_mushrooms[col].unique()) <= 1:
        print("Removing column {}, which only contains the value: {}".format(col, df_mushrooms[col].unique()[0]))
        
#%% DIFFERENT METHODS OF HANDLING MISSING VALUES
        
#Drop rows with missing values
print("Number of rows in total: {}".format(df_mushrooms.shape[0]))
print("Number of rows with missing values in column 'stalk-root': {}".format(df_mushrooms[df_mushrooms['stalk-root'] == '?'].shape[0]))
df_mushrooms_dropped_rows = df_mushrooms[df_mushrooms['stalk-root'] != '?']

#Drop col with more than X% missing vals
drop_percentage = 0.8

df_mushrooms_dropped_cols = df_mushrooms.copy(deep=True)
df_mushrooms_dropped_cols.loc[df_mushrooms_dropped_cols['stalk-root'] == '?', 'stalk-root'] = np.nan

for col in df_mushrooms_dropped_cols.columns.values:
    no_rows = df_mushrooms_dropped_cols[col].isnull().sum()
    percentage = no_rows / df_mushrooms_dropped_cols.shape[0]
    if percentage > drop_percentage:
        del df_mushrooms_dropped_cols[col]
        print("Column {} contains {} missing values. This is {} percent. Dropping this column.".format(col, no_rows, percentage))
        
#Fill missing vals with zeros
df_mushrooms_zerofill = df_mushrooms.copy(deep = True)
df_mushrooms_zerofill.loc[df_mushrooms_zerofill['stalk-root'] == '?', 'stalk-root'] = np.nan
df_mushrooms_zerofill.fillna(0, inplace=True)

#Backwards fill
df_mushrooms_bfill = df_mushrooms.copy(deep = True)
df_mushrooms_bfill.loc[df_mushrooms_bfill['stalk-root'] == '?', 'stalk-root'] = np.nan
df_mushrooms_bfill.fillna(method='bfill', inplace=True)

#Forward fill
df_mushrooms_ffill = df_mushrooms.copy(deep = True)
df_mushrooms_ffill.loc[df_mushrooms_ffill['stalk-root'] == '?', 'stalk-root'] = np.nan
df_mushrooms_ffill.fillna(method='ffill', inplace=True)

#%% ONE HOT ENCODING OF CATEGORICAL COLS (NOT THE BEST OPTION - BETTER TO CREATE BINARY COLUMNS BELOW)

def label_encode(df, columns):
    for col in columns:
        le = LabelEncoder()
        col_values_unique = list(df[col].unique())
        le_fitted = le.fit(col_values_unique)

        col_values = list(df[col].values)
        le.classes_
        col_values_transformed = le.transform(col_values)
        df[col] = col_values_transformed


df_mushrooms_ohe = df_mushrooms.copy(deep=True)
to_be_encoded_cols = df_mushrooms_ohe.columns.values
label_encode(df_mushrooms_ohe, to_be_encoded_cols)
display(df_mushrooms_ohe.head())

## Now lets do the same thing for the other dataframes
df_mushrooms_dropped_rows_ohe = df_mushrooms_dropped_rows.copy(deep = True)
df_mushrooms_zerofill_ohe = df_mushrooms_zerofill.copy(deep = True)
df_mushrooms_bfill_ohe = df_mushrooms_bfill.copy(deep = True)
df_mushrooms_ffill_ohe = df_mushrooms_ffill.copy(deep = True)

label_encode(df_mushrooms_dropped_rows_ohe, to_be_encoded_cols)
label_encode(df_mushrooms_zerofill_ohe, to_be_encoded_cols)
label_encode(df_mushrooms_bfill_ohe, to_be_encoded_cols)
label_encode(df_mushrooms_ffill_ohe, to_be_encoded_cols)

#%% BINARIZE CATEGORICAL COLS

def expand_columns(df, list_columns):
    for col in list_columns:
        colvalues = df[col].unique()
        for colvalue in colvalues:
            newcol_name = "{}_is_{}".format(col, colvalue)
            df.loc[df[col] == colvalue, newcol_name] = 1
            df.loc[df[col] != colvalue, newcol_name] = 0
    df.drop(list_columns, inplace=True, axis=1)


y_col = 'class'
to_be_expanded_cols = list(df_mushrooms.columns.values)
to_be_expanded_cols.remove(y_col)

df_mushrooms_expanded = df_mushrooms.copy(deep=True)
label_encode(df_mushrooms_expanded, [y_col])
expand_columns(df_mushrooms_expanded, to_be_expanded_cols)


## Now lets do the same thing for all other dataframes
df_mushrooms_dropped_rows_expanded = df_mushrooms_dropped_rows.copy(deep = True)
df_mushrooms_zerofill_expanded = df_mushrooms_zerofill.copy(deep = True)
df_mushrooms_bfill_expanded = df_mushrooms_bfill.copy(deep = True)
df_mushrooms_ffill_expanded = df_mushrooms_ffill.copy(deep = True)

label_encode(df_mushrooms_dropped_rows_expanded, [y_col])
label_encode(df_mushrooms_zerofill_expanded, [y_col])
label_encode(df_mushrooms_bfill_expanded, [y_col])
label_encode(df_mushrooms_ffill_expanded, [y_col])

expand_columns(df_mushrooms_dropped_rows_expanded, to_be_expanded_cols)
expand_columns(df_mushrooms_zerofill_expanded, to_be_expanded_cols)
expand_columns(df_mushrooms_bfill_expanded, to_be_expanded_cols)
expand_columns(df_mushrooms_ffill_expanded, to_be_expanded_cols)

#%% TRAIN ON EACH OF THE IMPUTATION METHODS TO SEE WHICH PERFORMS BEST

#Dictionary of dataframes to loop through
dict_dataframes = {
    "df_mushrooms_ohe": df_mushrooms_ohe,
    #"df_mushrooms_dropped_rows_ohe": df_mushrooms_dropped_rows_ohe,
    #"df_mushrooms_zerofill_ohe": df_mushrooms_zerofill_ohe,
    #"df_mushrooms_bfill_ohe": df_mushrooms_bfill_ohe,
    #"df_mushrooms_ffill_ohe": df_mushrooms_ffill_ohe,
    "df_mushrooms_expanded": df_mushrooms_expanded,
    #"df_mushrooms_dropped_rows_expanded": df_mushrooms_dropped_rows_expanded,
    #"df_mushrooms_zerofill_expanded": df_mushrooms_zerofill_expanded,
    #"df_mushrooms_bfill_expanded": df_mushrooms_bfill_expanded,
    #"df_mushrooms_ffill_expanded": df_mushrooms_ffill_expanded
}
 
#Dictionary of classifiers to loop over
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
    #"Gaussian Process": GaussianProcessClassifier()
}

def get_train_test(df, y_col, x_cols, ratio):

    mask = np.random.rand(len(df)) < train_test_ratio
    df_train = df[mask]
    df_test = df[~mask]
       
    X_train = df_train[x_cols].values
    X_test = df_test[x_cols].values

    Y_train = df_train[y_col].values
    Y_test = df_test[y_col].values

    return df_train, df_test, X_train, Y_train, X_test, Y_test


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

#%% EXECUTE PREVIOUS FUNCTIONS
    
y_col = 'class'
train_test_ratio = 0.7

for df_key, df in dict_dataframes.items():
    x_cols = list(df.columns.values)
    x_cols.remove(y_col)
    df_train, df_test, X_train, Y_train, X_test, Y_test = get_train_test(df, y_col, x_cols, train_test_ratio)
    dict_models = batch_classify(X_train, Y_train, X_test, Y_test, no_classifiers = 10, verbose=False)
    
    print()
    print(df_key)
    display_dict_models(dict_models)
    print("-------------------------------------------------------")
    
    
    