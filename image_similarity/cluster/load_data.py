#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 09:41:46 2019

@author: thomas
"""
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
