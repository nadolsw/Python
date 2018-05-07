import numpy as np
import pandas as pd
import csv as csv
import matplotlib
import unicodecsv
from sklearn import datasets
from __future__ import division

##############################################################

pd.options.display.max_columns
pd.set_option('display.max_columns', 25)
pd.set_option('expand_frame_repr', False)
   
#Load data into pandas dataframe directly
filepath = "C:\\Users\\nadolsw\\Desktop\\Python\\Udacity\\Intro to Data Analysis with Python\\enrollments.csv"
enrollment_df = pd.read_csv(filepath)

print(enrollment_df.head(5))
print(enrollment_df.tail(10))

#Create new DF and sort by join_date column
enrollment_df_by_jdate = enrollment_df.sort_values(by='join_date')
enrollment_df_by_jdate['join_date'].head(15)
enrollment_df_by_jdate[['join_date','cancel_date','days_to_cancel']].head(15)

##############################################################

df = enrollment_df

df.sort_index()
df.rank
df.ndim
df.shape #list table dimensions
print "Data Frame dimensions are: " + str(df.shape)

df.columns #list columns
df.info() #contents of the dataframe

df['status'].unique() #list all unique values of a column
df['status'].nunique() #count number of unique values in a column
df.apply(pd.Series.nunique) #count number of unique values for every column in dataframe

current = df[df['status'] == 'current'] #select all obs where status is current
iscurrent_series = df['status'] == "current" #create a T/F series based on whether status is current
df.ix[0] #select the first row of the dataframe

transpose = df.T #transpose dataframe

cancel_date = df['cancel_date'] #extract cancel_date series (one dim df)
dates = df[['join_date','cancel_date']] #create dataframe with only two columns

df.count() #number of non-missing values for each variable
df.isnull().sum() #count number of missing values for each variable
df['cancel_date'].isnull().sum() #number of missing values for single column cancel_date

null_data = df[df.isnull().any(axis=1)] #create a dataframe of all records with at least one mising value
null_cancel_date = df[df['cancel_date'].isnull()] #create dataframe with every obs where cancel_date is null

dtc='days_to_cancel'
df.sum() #sum all numeric columns
df[dtc].sum() #sum only days_to_cancel column
df[dtc].sum(skipna=False) #don't skip over missing values

df[dtc].min()
df[dtc].max()
df[dtc].mean()
df[dtc].median()
df[dtc].mode()
df[dtc].describe()
df[dtc].describe(percentiles=[0.05, 0.95])

df.mean() 
df.mean(axis=0) #mean of every numeric column in dataframe
df.mean(axis=1) #mean of every column for each row in dataframe

##############################################################

#define function to display objects before and after a change
class display(object):
    """Display HTML representation of multiple objects"""
    template = """<div style="float: left; padding: 10px;">
    <p style='font-family:"Courier New", Courier, monospace'>{0}</p>{1}
    </div>"""
    def __init__(self, *args):
        self.args = args

    def _repr_html_(self):
        return '\n'.join(self.template.format(a, eval(a)._repr_html_())
                         for a in self.args)

    def __repr__(self):
        return '\n\n'.join(a + '\n' + repr(eval(a))
                           for a in self.args)

##############################################################

nonmissing = df.dropna(axis='rows') #create a dataframe consisting of only non-missing values (drops any row where NaN is found in any column)
A=df.head()
B=nonmissing.head()        
display('A','B')

nonmissing_columns = df.dropna(axis='columns') #drop any column containing a NaN (missing) value
Z=nonmissing_columns.head()
display('A','Z')

test = pd.DataFrame([[np.nan, 2, np.nan, 0], [3, 4, np.nan, 1], [np.nan, np.nan, np.nan, 5]],columns=list('ABCD'))

drop_missing_row = test.dropna(axis='rows', how='all') #drop records where all column values are missing
display('test','drop_missing_row')

drop_missing_columns = test.dropna(axis='columns', how='all') #drop columns where all record values are missing
display('test','drop_missing_columns')

keep_rows_thresh = test.dropna(axis='rows', thresh=2) #keep records with at least 2 nonmissing column values
keep_cols_thresh = test.dropna(axis='columns', thresh=1) #keep columns with at least 1 nonmissing value

#Calculate number of missing obs to be removed
nmiss=df.cancel_date.isnull().sum() #number of records missing cancel_date value
nobs=len(df) #total number of records in original dataframe
pct_miss="{:.2%}".format(nmiss/nobs) #percent of records to be removed - properly formatted (need to first invoke 'from __future__ import division' on line 7 above )

print str(nmiss) + ' of ' + str(nobs) + ' obs are missing (' + str(pct_miss) + ') of data will be dropped'
drop_missing_cancel_date_records = df.dropna(subset=['cancel_date'])

##############################################################

fill_df_with_zeros = test.fillna(0) #fill missing values with zero across entire dataframe
fill_col_with_zeroes = test[['A','B']].fillna(100) #fill missing values for only columns A & B with 100
fill_with_mean = test.fillna(test.mean()) #fill all missing values with mean of column
fill_col_mean = test.fillna(test.mean()['B':'B']) #fill only column B with mean of column B

test.iloc[2,3] = np.nan #set specific cell D3 to missing

replace_missing_with_col_specific_values = test.fillna(value={'A':100,'B':0,'C':25,'D':75})

fill_with_diff_col_mean = test.fillna(test.D.mean()) #fill all missing cells with mean of column D

fill_down = test.fillna(method='ffill')
fill_up = test.fillna(method='bfill')

new_record = pd.DataFrame([[np.nan,np.nan,np.nan,np.nan]], columns=list('ABCD')) #create an array of missing values with same label names
add_empty_record = test.append(new_record, ignore_index=True) #append new record to existing dataframe
add_empty_record = add_empty_record.append(new_record, ignore_index=True) #append new record to existing dataframe

limit_fill_down = add_empty_record.fillna(method='ffill', limit=1) #limits the number of consecutive records to fill
display('add_empty_record','limit_fill_down')
##############################################################