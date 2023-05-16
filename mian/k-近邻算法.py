import numpy as np
from csv import reader
import pandas as pd
import matplotlib.pyplot as plt
import csv
import random
import math
import operator
#读取本地数据
#在函数中修改trainingSet和testSet，全局变量trainingSet和testSet也会发生改变:传的参数是引用，即直接检索的是地址
def loadDataset(filename,split,trainingSet=[],testSet=[]):
    with open(filename,'r') as csvfile:
        lines=csv.reader(csvfile)
        dataset=list(lines)
        for x in range(len(dataset)-1):
            for y in range(4):
                dataset[x][y]=float(dataset[x][y])
            if random.random()<split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])


#计算欧氏距离：
'''
先求得每对样本件的不同特征的差异值，
然后求差值的平方和，
然后再求这个和的平方根
'''
def EuclidDist(instance1,instance2,len):
    distance=0.0
    for x in  range(len):
        distance+=pow((instance1[x]-instance2[x]),2)
    return math.sqrt(distance)

#找位置点的邻居
def getNeighbors(trainSet,testInstance,k):
    distances=[]
    length=len(testInstance)-1
    for x in range(len(trainSet)):
        dist=EuclidDist(testInstance,trainSet[x],length)
        distances.append((trainSet[x],dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors=[]
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

#判断归属的函数getClass
'''
统计邻居的类别，使用投票决策进行判别
'''
def getClass(neighbors):
    classVotes={}
    for x in range(len(neighbors)):
        instance_class=neighbors[x][-1]
        if instance_class in classVotes:
            classVotes[instance_class]+=1
        else:
            classVotes[instance_class]=1
    #python的内置函数sorted（），原型是sorted（iterable，key，reverse）。
    #iterable：指定要排序的可迭代对象。本例中classVotes.items()返回可迭代的字典元素
    #key：指定取待排序的那一项进行排序
    #reverse：布尔变量，true是降序，false是升序（默认）
    sortedVotes=sorted(classVotes.items(),key=operator.itemgetter(1),reverse=True)
    return sortedVotes[0][0]


#模型评估
'''
评估的指标是，测试集合的预测类别与其真实类别的比率
'''
def getAccurcy(testSet,predictions):
    correct=0
    for x in range(len(testSet)):
        if(testSet[x][-1]==predictions[x]):
            correct+=1
    return (correct/float(len(testSet)))*100.0

def main():
    trainingSet=[]
    testSet=[]
    split=0.7
    loadDataset('D:\workspace\K近邻分类法\iris.csv',split,trainingSet,testSet)
    print('训练集样本数：' + repr(len(trainingSet)))
    print('测试集样本数：' + repr(len(testSet)))
    predictions=[]
    k=3
    #对预测集合元素进行预测
    for x in range(len(testSet)):
        #根据欧式距离（欧几里得）获取要进行预测的元素的neighbor
        neighbors=getNeighbors(trainingSet,testSet[x],k)
        #调用getClass函数，获取预测类别，然后存储
        result=getClass(neighbors)
        predictions.append(result)
        print('>预测='+repr(result)+',实际='+repr(testSet[x][-1]))
    #调用getAccuracy函数，对模型进行评估
    accuracy=getAccurcy(testSet,predictions)
    print('精确度为：'+repr(accuracy)+'%')


main()

