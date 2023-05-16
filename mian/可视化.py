import numpy as np
from csv import reader
import pandas as pd
import matplotlib.pyplot as plt
import csv
import random
import math
import operator
filename='D:\workspace\K近邻分类法\iris.csv'
df=pd.read_csv(filename,header=None)
X=df.iloc[0:150,[0,2]].values
plt.scatter(X[0:50,0],X[:50,1],color='blue',marker='x',label='setosa')
plt.scatter(X[50:100,0],X[50:100,1],color='red',marker='o',label='versicolor')
plt.scatter(X[100:150,0],X[100:150,1],color='green',marker='*',label='virginica')
plt.xlabel('petal width')
plt.ylabel('sepal length')
plt.legend(loc='upper left')
plt.show()

