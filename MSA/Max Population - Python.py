# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 13:03:08 2015

@author: William
"""

import csv
inp=open('C:\Users\William\Desktop\pop.csv','rb')
reader=csv.reader(inp)
header=reader.next()

max_pop=0
max_city='Unknown'
max_state='Unknown'

for row in reader:
    pop=int(row[11])
    if row[6]!=row[7]:
        if pop>max_pop:
            max_pop=pop
            max_city=row[6]
            max_state=row[7]
inp.close()
print max_city, max_state, max_pop

