#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri June 25 10:18:20 2020

@author: berkayinan
"""

#2.import(kutuphaneler)
import pandas as pd
import numpy as np
import matplotlib as plt



#2.1. Veri Yukleme

veriler = pd.read_csv("maaslar.csv")

#dataframe (dilimleme/slice)
x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]
#numpy dizi (array) dönüşümü
X = x.values
Y = y.values

#linear regression
#doğrusal model
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

#polynomial regression
#doğrusal olmayan(nonlinear model) oluşturma
#2.dereceden polinom
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree= 2)
x_poly = poly_reg.fit_transform(X)
#print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)

#4.dereceden polinom
poly_reg3 = PolynomialFeatures(degree= 4)
x_poly3 = poly_reg3.fit_transform(X)
#print(x_poly)
lin_reg3 = LinearRegression()
lin_reg3.fit(x_poly3,y)

#göörselleştirme
#olmazsa x.values y.values yapabilirsin
import matplotlib.pyplot as plt
plt.scatter(X,Y,color='red')
plt.plot(x, lin_reg.predict(X), color='blue')
plt.show()

plt.scatter(X,Y, color='red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)), color='blue')
plt.show()

plt.scatter(X,Y, color='red')
plt.plot(X,lin_reg3.predict(poly_reg3.fit_transform(X)), color='blue')
plt.show()

#tahminler

#print(lin_reg.predict(11))
#print(lin_reg.predict(6.6))

#rint(lin_reg2.predict(poly_reg.fit_transform(11)))
#print(lin_reg2.predict(poly_reg.fit_transform(6.6)))

#verilerin olceklenmesi 
from sklearn.preprocessing import StandardScaler

sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(X)
sc2 = StandardScaler()
y_olcekli = sc2.fit_transform(Y)  

from sklearn.svm import SVR

svr_reg = SVR(kernel='rbf')
svr_reg.fit(x_olcekli,y_olcekli)

plt.scatter(x_olcekli,y_olcekli, color='red')
plt.plot(x_olcekli,svr_reg.predict(x_olcekli),color='blue')

#print(svr_reg.predict(11))



