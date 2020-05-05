#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 14:35:10 2019

@author: thomas
"""
import simplejson as json
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv('pria_wanita_vector.csv')

array_pria = []
array_wanita = []
for i in range(len(df)):
    p = [1] + json.loads(df['pria'][i])
    array_pria.append(p)
    w = [1] + json.loads(df['wanita'][i])
    array_wanita.append(w)

zero = [1] + list(np.zeros(len(array_pria[0])-1))

array = array_pria + array_wanita + [zero]

# Gt
array = np.array(array)
Gt = array.transpose()

# G
G = array
# GtG
GtG = np.matmul(Gt, G)
# d
d = []
for i in range(len(array_pria)):
    d.append([1])
for i in range(len(array_wanita)):
    d.append([-1])
d.append([0])

# TODO!
d = np.array(d)
# Gtd
Gtd = np.matmul(Gt, d)
# invGtG
invGtG = np.linalg.inv(np.matmul(Gt, G))

# m
m = np.matmul(np.matmul(invGtG, Gt), d)

# label = d[i][0]

# Model
def predict(m, data):
    f = m[0][0]
    for i in range(len(m)):
        if i == 0:
            pass
        else:
            data_i = m[i]*data[i-1]
            f = f + data_i
    return f


data_x = array_pria[0][1:]
label_asli_x = d[0][0]
f = predict(m, data_x)

# =============================================================================
# LR
# =============================================================================
array_pria_train = []
array_wanita_train = []
for i in range(len(df)):
    p_train = json.loads(df['pria'][i])
    array_pria_train.append(p_train)
    w_train = json.loads(df['wanita'][i])
    array_wanita_train.append(w_train)

data_train = array_pria[:len(array_pria)-5] \
    + array_wanita[:len(array_wanita)-5]

data_test = array_pria[len(array_pria)-5:] \
    + array_wanita[len(array_wanita)-5:]

label_train = []
for i in range(len(array_pria[:len(array_pria)-5])):
    label_train.append(1)
for i in range(len(array_wanita[:len(array_wanita)-5])):
    label_train.append(-1)

label_test = []
for i in range(len(array_pria[len(array_pria)-5:])):
    label_test.append(1)
for i in range(len(array_wanita[len(array_wanita)-5:])):
    label_test.append(-1)


model = LinearRegression().fit(np.array(data_train), np.array(label_train))
coef = model.coef_
intercept = model.intercept_

score = model.score(np.array(data_test), np.array(label_test))

model.predict(np.array([data_test[0]]))


#[-1.03538707,  0.30174989,  0.40950379, -0.85583809, -0.72864862,
#-0.796675  , -0.41913839, -0.72949341, -0.5505873 , -0.07020929]