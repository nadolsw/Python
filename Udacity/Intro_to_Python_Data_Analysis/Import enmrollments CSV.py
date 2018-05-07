#CSV must be located in same directory as .py script
#.py script must be saved before it can be executed/compiled

import numpy as np
import pandas as pd
import csv as csv
import matplotlib
import unicodecsv
from sklearn import datasets

##########################################
# import csv
# file='enrollments.csv'

# with open(file,'rb') as f:
# 	reader=csv.reader(f)
# 	for row in reader:
# 		print row

# f.close()
##########################################

##########################################
# import unicodecsv
# enrollments=[]

# f=open('enrollments.csv','rb')
# reader=unicodecsv.DictReader(f)

# for row in reader:
# 	enrollments.append(row)

# f.close()
# enrollments[0]
##########################################

##########################################
#Alternative method
# import unicodecsv
# with open('enrollments.csv','rb') as f:
# 	reader=unicodecsv.DictReader(f)
# 	enrollments=list(reader)

# print(enrollments)
##########################################


##########################################
#define read file function
# import unicodecsv
# def read_csv(filename):
# 	 with open(filename,'rb') as f:
# 	 	reader=unicodecsv.DictReader(f)
# 	 	return list(reader)

# enrollments=read_csv('enrollments.csv')
# daily_engagement=read_csv('daily_engagement.csv')
# project_submissions=read_csv('project_submissions.csv')

# print(enrollments)
##########################################


##########################################
# import unicodecsv
# engagement=[]
# submissions=[]

# eng=open('daily_engagement.csv','rb')
# sub=open('project_submissions.csv','rb')

# read_eng=unicodecsv.DictReader(eng)
# read_sub=unicodecsv.DictReader(sub)

# for row in read_eng:
# 	engagement.append(row)
# eng.close()

# for row in read_sub:
# 	submissions.append(row)
# sub.close()

# print(engagement[0])
# print(submissions[0])
##########################################

#Load data into pandas dataframe
enrollment = pd.read_csv('enrollments.csv')
DF = pd.DataFrame(enrollment) 

pd.options.display.max_columns
pd.set_option('display.max_columns', 25)
pd.set_option('expand_frame_repr', False)

print(DF.head())
print(DF.tail())

print "Data Frame dimensions are: " + str(DF.shape)

print 'Hello'