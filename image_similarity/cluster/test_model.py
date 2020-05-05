#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 09:02:23 2019

@author: thomas
"""
import pandas as pd

from sklearn.externals import joblib
# joblib.dump(kmeans, 'model_j.pkl')
model_j = joblib.load('model_j.pkl') 

data = pd.read_csv('jawa_negro_vector.csv')


import pandas as pd
import simplejson as json

df = pd.read_csv('pria_wanita_vector.csv')

p = df['pria'][0]
p = json.loads(p)
#print(type(p))

negro = []
for i in range(4):
    negro.append(json.loads(df['pria'][i]))

jawa = [json.loads(df['wanita'][len(df['wanita'])-1]),
        json.loads(df['wanita'][len(df['wanita'])-1])]



n = model_j.predict(negro)
j = model_j.predict(jawa)


