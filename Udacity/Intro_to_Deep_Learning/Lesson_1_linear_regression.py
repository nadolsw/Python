# -*- coding: utf-8 -*-
"""
Created on Sun Apr 01 22:29:31 2018

@author: nadolsw
"""
import csv
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt


filepath="C:\\Users\\nadolsw\\Desktop\\Python\\Udacity\\Intro to Deep Learning\\brain_body.csv"
df = pd.read_csv(filepath)

x = df[['BRAIN']]
y = df[['BODY']]

reg = linear_model.LinearRegression()
reg.fit(x,y)

plt.scatter(x,y)
plt.plot(x,reg.predict(x))
plt.show()

