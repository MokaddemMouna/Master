
# -*- coding:Latin-1 -*-
############################################################################################
############################################################################################
# 
# Training Binary Circuits For Clasification and High Level Representation
#
############################################################################################
############################################################################################
# Abstract
#
# I present here the general algorithm for constructing booleen circuit in machine learning. 
# For supervise learning, with multiple categories and unsupervise learning with autoencoder.


# Package

import operator
import random
from random import shuffle
import math
import pickle
import numpy
#import scipy
from sklearn import svm
import copy
import multiprocessing
import csv
#import time
from gmpy2 import *
import functools 
import graphics 
import sys
import gc
import math as mth
from itertools import *
from datetime import datetime

sys.setrecursionlimit(10000)

def Transpose(l):
    return list(zip(*l))


def PosMax(l):
    m = max(l)
    tmp = [i for i, j in enumerate(l) if j == m]
    return tmp[0]

def H(x):
    if (x == 0.0 or x == 1.0):
        return 0
    else:
        return (-x * log(x) - (1 - x) * log(1 - x)) / log(2)

##############################################################################
##############################################################################
##############################################################################
# Create Data MOUNA
# MNIST
def GenDataMNIST_2cat_brut(trainFile,testFile,cat1,cat2,nbbits,nbTrain,nbTest,nbValid):
    
    def readMNIST(filename):
        a = []
        with open(filename,"r") as file:
            for line in file:
                line = (line.strip()).split()
                line = list(map(lambda x:int(float(x) * 256),line)) #float(x) * 256 tu l'ecris toujours sur 8 bits alors ?
                line[784] = int(line[784] / 256) #pourquoi diviser sur 256 ?
                a.append(line)
        return a

    #prend un entier, le transforme,d'abord, en son eq binaire sous format de
    #liste par exple x= 5 -> [0,0,0,0,0,1,0,1] et ?a fin elle retourne une
    #liste avec longueur nbbits
    def f(x):
        lb = [int(y) for y in bin(x)[2:]] #par exple si x=5, lb = [1,0,1]
        lbb = [0 for _ in range(8 - len(lb))] 
        lbb.extend(lb)
        return lbb[0:nbbits]

    a = readMNIST(trainFile)    # "..\..\..\data sets\mnist basic\mnist_train.amat"
    b = readMNIST(testFile)
    a.extend(b)
    l = len(a)
    
    print("MNIST file loaded.")
    nbimages = len(a)
    print("Total number of images: " + str(nbimages))

    #cat1_2List = []
    #for image in a:
    #    s = image[784]
    #    if s in cat1 or s in cat2:
    #        cat1_2List.append(image)
    #print("nb of examples of the two categories = ",len(cat1_2List))
    
    #if (nbTrain + nbTest+nbValid) > len(cat1_2List):
    #    print("nbtrain or/and nbTest or/and nbValid should be lower")
    #else:
    #train_test_valid = random.sample(cat1_2List,nbTrain + nbTest + nbValid)
    train_test_valid = random.sample(a,nbTrain + nbTest + nbValid)
    train = train_test_valid[0:nbTrain]
    valid = train_test_valid[nbTrain:nbTrain + nbValid]
    tot_train = train_test_valid[0:nbTrain + nbValid]
    test = train_test_valid[nbTrain + nbValid:nbTrain + nbValid + nbTest]
    
    def dataStructure(a):
        Data = []
        Target = []
        for image in a:
        
            s = image[28 * 28]
        
            bimage0 = list(map(f,image[0:784]))
            bimage = [val for subl in bimage0 for val in subl]
            Data.append(bimage)
            #if s in cat1:
            if s == cat1:
               Target.append(0)
            else:
               Target.append(1)
           
        N = len(Data)
        print("Images from the 2 categories: " + str(N))
        return Data, Target

    TrainData, TrainTarget = dataStructure(train)
    TestData, TestTarget = dataStructure(test)
    ValidateData, ValidateTarget = dataStructure(valid)
    TotTrainData, TotTrainTarget = dataStructure(tot_train)

    return TrainData,TrainTarget,TestData,TestTarget, ValidateData,ValidateTarget, TotTrainData, TotTrainTarget

def GenDataMNIST_10cat_brut(trainFile,testFile,CategoryList,nbbits,nbTrain,nbTest,nbValid):
    
    def readMNIST(filename):
        a = []
        with open(filename,"r") as file:
            for line in file:
                line = (line.strip()).split()
                line = list(map(lambda x:int(float(x) * 256),line)) #float(x) * 256 tu l'ecris toujours sur 8 bits alors ?
                line[784] = int(line[784] / 256) #pourquoi diviser sur 256 ?
                a.append(line)
        return a

    #prend un entier, le transforme,d'abord, en son eq binaire sous format de
    #liste par exple x= 5 -> [0,0,0,0,0,1,0,1] et ?a fin elle retourne une
    #liste avec longueur nbbits
    def f(x):
        lb = [int(y) for y in bin(x)[2:]] #par exple si x=5, lb = [1,0,1]
        lbb = [0 for _ in range(8 - len(lb))] 
        lbb.extend(lb)
        return lbb[0:nbbits]

    a = readMNIST(trainFile)    # "..\..\..\data sets\mnist basic\mnist_train.amat"
    b = readMNIST(testFile)
    a.extend(b)
    
    print("MNIST file loaded.")
    nbimages = len(a)
    print("Total number of images: " + str(nbimages))

    train_test_valid = random.sample(a,nbTrain + nbTest + nbValid)
    train = train_test_valid[0:nbTrain]
    valid = train_test_valid[nbTrain:nbTrain + nbValid]
    tot_train = train_test_valid[0:nbTrain + nbValid]
    test = train_test_valid[nbTrain + nbValid:nbTrain + nbValid + nbTest]
        
    
    def dataStructure(a):
        Data = []
        Target = [[] for _ in range(len(a))]
        k = 0
        for image in a:
            mark = False
            s = image[28 * 28]
        
            bimage0 = list(map(f,image[0:784])) #on passe par bimage0 pour pouvoir appliquer f et ecrire les features sur le
                                                #nombre de bits nbbits

            bimage = [val for subl in bimage0 for val in subl]
            Data.append(bimage)
            for i in CategoryList:
                if mark == False:
                    if s == i:
                        mark = True
                        for c in range(10):
                            if c == i:
                                Target[k].append(1)
                            else:
                                Target[k].append(0)
                        k = k + 1
                else:
                    break
            
        N = len(Data)
        print("Images from the 2 categories: " + str(N))
        return Data, Target

    TrainData, TrainTarget = dataStructure(train)
    TestData, TestTarget = dataStructure(test)
    ValidateData, ValidateTarget = dataStructure(valid)
    TotTrainData, TotTrainTarget = dataStructure(tot_train)

    return TrainData,TrainTarget,TestData,TestTarget, ValidateData,ValidateTarget, TotTrainData, TotTrainTarget
    #return TrainData
     

def GenDataMNIST_BW_2Cat(FileDump, Cat1, Cat2, Noise, Type, NbTrain, NbTest, NbValid):
    if Type == "basic":
        trainFile = "..\..\..\..\Visual Studio 2013\Projects\MNIST basic\mnist_train.amat"
        testFile = "..\..\..\..\Visual Studio 2013\Projects\MNIST basic\mnist_test.amat"
    elif Type == "convex":
        trainFile = "..\..\..\..\Visual Studio 2013\Projects\LISA convex\convex_train.amat"
        testFile = "..\..\..\..\Visual Studio 2013\Projects\LISA convex\convex_test.amat" 
    elif Type == "random":
        trainFile = "..\..\..\Projects\MNIST random background\mnist_background_random_train.amat"
        testFile = "..\..\..\Projects\MNIST random background\mnist_background_random_test.amat"
    else:
        trainFile = "..\..\..\Projects\MNIST image background\mnist_background_images_train.amat"
        testFile = "..\..\..\Projects\MNIST image background\mnist_background_images_test.amat"
    
    #???
    #File = "..\..\..\data sets\mnist basic\mnist_small.amat"
    TrainData, TrainTarget, TestData, TestTarget, ValidateData, ValidateTarget, TotTrainData, TotTrainTarget  = GenDataMNIST_2cat_brut(trainFile,testFile,Cat1,Cat2,1, NbTrain ,NbTest, NbValid)

    DataT = BinaryDataSet("This is a test", TotTrainData, TotTrainTarget, TestData, TestTarget)

    DataV = BinaryDataSet("This is a validation", TrainData, TrainTarget, ValidateData, ValidateTarget)

    f = open(FileDump + "V.dump", "wb")
    #pickle.dump(DataV,f)
    pickle.dump(DataV,f,protocol=2)
    f.close()

    f = open(FileDump + "T.dump", "wb")
    #pickle.dump(DataT,f)
    pickle.dump(DataT,f,protocol=2)
    f.close()


def GenDataMNIST_BW_10Cat(FileDump, CategoryList, Noise, Type, NbTrain, NbTest, NbValid):
    if Type == "basic":
        #trainFile = "..\..\..\..\Visual Studio 2013\Projects\MNIST basic\mnist_small.amat"
        trainFile = "..\..\..\..\Visual Studio 2013\Projects\MNIST basic\mnist_train.amat"
        testFile = "..\..\..\..\Visual Studio 2013\Projects\MNIST basic\mnist_test.amat"
        #trainFile = "..\..\..\Visual Studio 2013\Projects\MNIST_basic\MNIST basic\mnist_train.amat"
        #testFile = "..\..\..\Visual Studio 2013\Projects\MNIST_basic\MNIST basic\mnist_test.amat"
        #trainFile = "..\..\..\data sets\mnist basic\mnist_train.amat"
        #testFile = "..\..\..\data sets\mnist basic\mnist_test.amat" 
        #File = "..\..\..\data sets\mnist basic\mnist_train.amat"
    elif Type == "convex":
        trainFile = "..\..\..\Visual Studio 2013\Projects\LISA_convex\LISA convex\convex_train.amat"
        testFile = "..\..\..\Visual Studio 2013\Projects\LISA_convex\LISA convex\convex_test.amat" 
        #trainFile = "..\..\..\data sets\LISA convex\convex_train.amat"
        #testFile = "..\..\..\data sets\LISA convex\convex_test.amat" 
    elif Type == "random":
        trainFile = "..\..\..\Projects\MNIST random background\mnist_background_random_train.amat"
        testFile = "..\..\..\Projects\MNIST random background\mnist_background_random_test.amat"
    else:
        trainFile = "..\..\..\Projects\MNIST image background\mnist_background_images_train.amat"
        testFile = "..\..\..\Projects\MNIST image background\mnist_background_images_test.amat"
    
    #???
    #File = "..\..\..\data sets\mnist basic\mnist_small.amat"
    TrainData, TrainTarget, TestData, TestTarget, ValidateData, ValidateTarget, TotTrainData, TotTrainTarget  = GenDataMNIST_10cat_brut(trainFile,testFile,CategoryList,1, NbTrain ,NbTest, NbValid)

    DataT = DataSet("This is a test", TotTrainData, TotTrainTarget, TestData, TestTarget)

    DataV = DataSet("This is a validation", TrainData, TrainTarget, ValidateData, ValidateTarget)
    

    f = open(FileDump + "V.dump", "wb")
    #pickle.dump(DataV,f)
    pickle.dump(DataV,f,protocol=2)
    f.close()

    f = open(FileDump + "T.dump", "wb")
    #pickle.dump(DataT,f)
    pickle.dump(DataT,f,protocol=2)
    f.close()

def GenDataCIFAR_2cat_brut(batch1,batch2,batch3,batch4,batch5,test_batch,nbbits,nbTrain,nbTest,nbValid,cat1,cat2):

    def unpickle(file):
        fo = open(file, 'rb')
        dict = pickle.load(fo,encoding='latin1')
        fo.close()
        elt = dict['data'].tolist()
        lb = dict['labels']
        return elt,lb

    def f(x):
        lb = [int(y) for y in bin(x)[2:]] #par exple si x=5, lb = [1,0,1]
        lbb = [0 for _ in range(8 - len(lb))] 
        lbb.extend(lb)
        return lbb[0:nbbits]

    data1,label1 = unpickle(batch1)
    data2,label2 = unpickle(batch2)
    data3,label3 = unpickle(batch3)
    data4,label4 = unpickle(batch4)
    data5,label5 = unpickle(batch5)
    testData,testLabel = unpickle(test_batch)

    Data = []
    Data.extend(data1)
    Data.extend(data2)
    Data.extend(data3)
    Data.extend(data4)
    Data.extend(data5)
    Data.extend(testData)

    Label = []
    Label.extend(label1)
    Label.extend(label2)
    Label.extend(label3)
    Label.extend(label4)
    Label.extend(label5)
    Label.extend(testLabel)

    print("CIFAR file loaded.")
    nbimages = len(Data)
    print("Total number of train images: " + str(nbimages))

    cat12 = []
    cat12_label = []
    for i in range(len(Label)):
        if Label[i] == cat1 or Label[i] == cat2:
            cat12.append(Data[i])
            cat12_label.append(Label[i])

    train = cat12[0:nbTrain]
    valid = cat12[nbTrain:nbTrain + nbValid]
    tot_train = cat12[0:nbTrain + nbValid]
    test = cat12[nbTrain + nbValid:nbTrain + nbValid + nbTest]

    trainTarget = cat12_label[0:nbTrain]
    validTarget = cat12_label[nbTrain:nbTrain + nbValid]
    tot_trainTarget = cat12_label[0:nbTrain + nbValid]
    testTarget = cat12_label[nbTrain + nbValid:nbTrain + nbValid + nbTest]

    def dataStructure(a,b):
        data = []
        target = []
        for w in range(len(a)):
        
            s = b[w]
        
            bimage0 = list(map(f,a[w]))
            bimage = [val for subl in bimage0 for val in subl]
            data.append(bimage)
            #if s in cat1:
            if s == cat1:
               target.append(0)
            else:
               target.append(1)
           
        N = len(data)
        print("Images from the 2 categories: " + str(N))
        return data, target

    TrainData, TrainTarget = dataStructure(train,trainTarget)
    TestData, TestTarget = dataStructure(test,testTarget)
    ValidateData, ValidateTarget = dataStructure(valid,validTarget)
    TotTrainData = TrainData
    TotTrainData.extend(ValidateData)
    TotTrainTarget = TrainTarget
    TotTrainTarget.extend(ValidateTarget)

    return TrainData,TrainTarget,TestData,TestTarget, ValidateData,ValidateTarget, TotTrainData, TotTrainTarget

def GenDataCIFAR_BW_2Cat(FileDump,nbbits, NbTrain, NbTest, NbValid,cat1,cat2):

    File1 = "..\..\..\..\Visual Studio 2013\Projects\cifar-10-batches-py\data_batch_1"
    File2 = "..\..\..\..\Visual Studio 2013\Projects\cifar-10-batches-py\data_batch_2"
    File3 = "..\..\..\..\Visual Studio 2013\Projects\cifar-10-batches-py\data_batch_3"
    File4 = "..\..\..\..\Visual Studio 2013\Projects\cifar-10-batches-py\data_batch_4"
    File5 = "..\..\..\..\Visual Studio 2013\Projects\cifar-10-batches-py\data_batch_5"
    File6 = "..\..\..\..\Visual Studio 2013\Projects\cifar-10-batches-py\\test_batch"

    TrainData, TrainTarget, TestData, TestTarget, ValidateData, ValidateTarget, TotTrainData, TotTrainTarget  = GenDataCIFAR_2cat_brut(File1,File2,File3,File4,File5,File6,nbbits,NbTrain,NbTest,NbValid,cat1,cat2)
    
    
    DataT = BinaryDataSet("This is a test", TotTrainData, TotTrainTarget, TestData, TestTarget)

    DataV = BinaryDataSet("This is a validation", TrainData, TrainTarget, ValidateData, ValidateTarget)
    

    f = open(FileDump + "V.dump", "wb")
    pickle.dump(DataV,f,protocol=2)
    f.close()

    f = open(FileDump + "T.dump", "wb")
    pickle.dump(DataT,f,protocol=2)
    f.close()


def GenDataCIFAR_10cat_brut(batch1,batch2,batch3,batch4,batch5,test_batch,nbbits,nbTrain,nbTest,nbValid):

    def unpickle(file):
        fo = open(file, 'rb')
        dict = pickle.load(fo,encoding='latin1')
        fo.close()
        elt = dict['data'].tolist()
        lb = dict['labels']
        return elt,lb

    def f(x):
        lb = [int(y) for y in bin(x)[2:]] #par exple si x=5, lb = [1,0,1]
        lbb = [0 for _ in range(8 - len(lb))] 
        lbb.extend(lb)
        return lbb[0:nbbits]

    data1,label1 = unpickle(batch1)
    data2,label2 = unpickle(batch2)
    data3,label3 = unpickle(batch3)
    data4,label4 = unpickle(batch4)
    data5,label5 = unpickle(batch5)
    testData,testLabel = unpickle(test_batch)

    Data = []
    Data.extend(data1)
    Data.extend(data2)
    Data.extend(data3)
    Data.extend(data4)
    Data.extend(data5)
    Data.extend(testData)

    Label = []
    Label.extend(label1)
    Label.extend(label2)
    Label.extend(label3)
    Label.extend(label4)
    Label.extend(label5)
    Label.extend(testLabel)

    print("CIFAR file loaded.")
    nbimages = len(Data)
    print("Total number of train images: " + str(nbimages))

    train = Data[0:nbTrain]
    valid = Data[nbTrain:nbTrain + nbValid]
    tot_train = Data[0:nbTrain + nbValid]
    test = Data[nbTrain + nbValid:nbTrain + nbValid + nbTest]

    trainTarget = Label[0:nbTrain]
    validTarget = Label[nbTrain:nbTrain + nbValid]
    tot_trainTarget = Label[0:nbTrain + nbValid]
    testTarget = Label[nbTrain + nbValid:nbTrain + nbValid + nbTest]

    def dataStructure(a,b):
        data = []
        target = [[] for _ in range(len(a))]
        k = 0
        for w in range(len(a)):
            mark = False
            s = b[w]
        
            bimage0 = list(map(f,a[w])) #on passe par bimage0 pour pouvoir appliquer f et ecrire les features sur le
                                                #nombre de bits nbbits

            bimage = [val for subl in bimage0 for val in subl]
            data.append(bimage)
            for i in range(10):
                if mark == False:
                    if s == i:
                        mark = True
                        for c in range(10):
                            if c == i:
                                target[k].append(1)
                            else:
                                target[k].append(0)
                        k = k + 1
                else:
                    break
            
        N = len(data)
        print("Images from the 10 categories: " + str(N))
        return data, target

    TrainData, TrainTarget = dataStructure(train,trainTarget)
    TestData, TestTarget = dataStructure(test,testTarget)
    ValidateData, ValidateTarget = dataStructure(valid,validTarget)
    TotTrainData = TrainData
    TotTrainData.extend(ValidateData)
    TotTrainTarget = TrainTarget
    TotTrainTarget.extend(ValidateTarget)

    return TrainData,TrainTarget,TestData,TestTarget, ValidateData,ValidateTarget, TotTrainData, TotTrainTarget


def GenDataCIFAR_BW_10Cat(FileDump,nbbits, NbTrain, NbTest, NbValid):

    File1 = "..\..\..\..\Visual Studio 2013\Projects\cifar-10-batches-py\data_batch_1"
    File2 = "..\..\..\..\Visual Studio 2013\Projects\cifar-10-batches-py\data_batch_2"
    File3 = "..\..\..\..\Visual Studio 2013\Projects\cifar-10-batches-py\data_batch_3"
    File4 = "..\..\..\..\Visual Studio 2013\Projects\cifar-10-batches-py\data_batch_4"
    File5 = "..\..\..\..\Visual Studio 2013\Projects\cifar-10-batches-py\data_batch_5"
    File6 = "..\..\..\..\Visual Studio 2013\Projects\cifar-10-batches-py\\test_batch"

    TrainData, TrainTarget, TestData, TestTarget, ValidateData, ValidateTarget, TotTrainData, TotTrainTarget  = GenDataCIFAR_10cat_brut(File1,File2,File3,File4,File5,File6,nbbits,NbTrain,NbTest,NbValid)
    
    
    DataT = DataSet("This is a test", TotTrainData, TotTrainTarget, TestData, TestTarget)

    DataV = DataSet("This is a validation", TrainData, TrainTarget, ValidateData, ValidateTarget)
    

    f = open(FileDump + "V.dump", "wb")
    pickle.dump(DataV,f,protocol=2)
    f.close()

    f = open(FileDump + "T.dump", "wb")
    pickle.dump(DataT,f,protocol=2)
    f.close()


##############################################################################
##############################################################################
##############################################################################
# Create Data

###############

def GenTwonormalElement(mu,sigma,nbel):

    def f(x):
        if x > 2 ** 16 - 1:
            x = 2 ** 16 - 1
        if x < 0:
            x = 0
        lb = [int(y) for y in bin(x)[2:]]
        lbb = [0 for _ in range(16 - len(lb))]
        lbb.extend(lb)
        return lbb


    item = [int(round(random.gauss(mu,sigma))) for _ in range(nbel)]
    item = list(map(f,item))
    item = [val for subl in item for val in subl]

    return item



def GenTwoNormal(File, Text, NbTrain, NbTest, NbValid, nbel, mu0,sigma0, mu1,sigma1):

    def gen(cat):
        if cat == 0 :
            Ex = GenTwonormalElement(mu0,sigma0,nbel)
        else:
            Ex = GenTwonormalElement(mu1,sigma1,nbel)
        return Ex

    TrainTarget = [random.randint(0,1) for _ in range(NbTrain)]
    TrainData = [gen(cat) for cat in TrainTarget]

    TestTarget = [random.randint(0,1) for _ in range(NbTest)]
    TestData = [gen(cat) for cat in TestTarget]

    ValidateTarget = [random.randint(0,1) for _ in range(NbValid)]
    ValidateData = [gen(cat) for cat in ValidateTarget]

    DataT = DataSet(Text, TrainData, TrainTarget, TestData, TestTarget)
    DataV = DataSet(Text, TrainData, TrainTarget, ValidateData, ValidateTarget)

    f = open(File + "T.dump", "wb")
    #pickle.dump(DataT,f)
    pickle.dump(DataT,f,protocol=2)
    f.close()

    f = open(File + "V.dump", "wb")
    #pickle.dump(DataV,f)
    pickle.dump(DataV,f,protocol=2)
    f.close()
    
    return DataT, DataV

# two cubes
# 32 by 32 images with one (10 by 10) cube or (one 8 by 8 and 6 by 6)
def GenTwoCubesDataSet(File, Text, NbTrain, NbTest, NbValid, Noise):

    def square(Image, px,py,size):
        for i in range(size):
            for j in range(size):
                Image[(px + i) * 32 + (py + j)] = 1 - Image[(px + i) * 32 + (py + j)]

    def gen(cat):
        Image = [0 for i in range(32 ** 2)]
        if cat == 0 :
            px = random.randint(0, 32 - 15)
            py = random.randint(0, 32 - 15)
            square(Image, px,py,15)
        else:
            px = random.randint(0, 32 - 12)
            py = random.randint(0, 32 - 12)
            square(Image, px,py,12)
            px = random.randint(0, 32 - 9)
            py = random.randint(0, 32 - 9)
            square(Image, px,py,9)
        return Image

    TrainTarget = [random.randint(0,1) for _ in range(NbTrain)]
    TrainData = [gen(cat) for cat in TrainTarget]

    TestTarget = [random.randint(0,1) for _ in range(NbTest)]
    TestData = [gen(cat) for cat in TestTarget]

    ValidateTarget = [random.randint(0,1) for _ in range(NbValid)]
    ValidateData = [gen(cat) for cat in ValidateTarget]

    AddNoise(TrainData, Noise)
    AddNoise(TestData, Noise)
    AddNoise(ValidateData, Noise)

    DataT = DataSet(Text, TrainData, TrainTarget, TestData, TestTarget)
    DataV = DataSet(Text, TrainData, TrainTarget, ValidateData, ValidateTarget)

    f = open(File + "T.dump", "wb")
    #pickle.dump(DataT,f)
    pickle.dump(DataT,f,protocol=2)
    f.close()

    f = open(File + "V.dump", "wb")
    #pickle.dump(DataV,f)
    pickle.dump(DataV,f,protocol=2)
    f.close()

    return DataT, DataV

# add noise to a bi dimentionnal table of bit
# Data is a binaru table of integer with values 0 or 1
# level is the level of noise.  Level = 0.03 mean 3% of noise added.
def AddNoise(Data, Level):
    for li in Data:
        for b in range(len(li)):
            if random.random() < Level:
                li[b] = 1 - li[b]

                  
def GenWhiteNoiseDataSet(NbTrain,NbTest, Noise):
    def gen(cat):
        
        if cat == 0 :
            Image = [0 for i in range(16 ** 2)]
        else:
            Image = [1 for i in range(16 ** 2)]
        return Image

    TrainTarget = [random.randint(0,1) for _ in range(NbTrain)]
    TrainData = [gen(cat) for cat in TrainTarget]

    TestTarget = [random.randint(0,1) for _ in range(NbTrain)]
    TestData = [gen(cat) for cat in TestTarget]

    AddNoise(TrainData, Noise)
    AddNoise(TestData, Noise)

    Data = DataSet("White noise.", TrainData, TrainTarget, TestData, TestTarget)

    return Data

def AfficheTextImage(Image,px,py):
    dim = int(math.sqrt(len(Image)))
    for i in range(dim):
        for j in range(dim):
            pt = graphics.Point(j + px, i + py)
            if Image[i * dim + j] == 0:
                pt.setFill("black")
            else:
                pt.setFill("white")
            pt.draw(win)
        
class BinaryDataSet:
    def __init__(self, NameString, TrainData, TrainTarget, TestData, TestTarget):
        self.NameString = NameString                    # information on the data, name, etc.
        self.NbFeatures = len(TrainData[0])             # number of bit of each example
        self.NbTrain = len(TrainData)                   # number of training examples
        self.TrainData = numpy.asarray(TrainData)       # scikitlearning SVC train data
        self.TrainTarget = numpy.asarray(TrainTarget)   # scikitlearning SVC train target
        self.NbTest = len(TestData)                     # number of test examples
        self.TestData = numpy.asarray(TestData)         # scikitlearning SVC test data
        self.TestTarget = numpy.asarray(TestTarget)     # scikitlearning SVC test target
        
        self.Features = []                              # list of features for our algorithms (Train and
                                                        # test)
        self.Objective = []                             # target of our algorithm

        TrainDataT = Transpose(TrainData)
        TestDataT = Transpose(TestData)
        

        self.Features = [  Feature(TrainDataT[i],  TestDataT[i],i) for i in range(self.NbFeatures)   ]
        self.Objective = ObjectiveClass(TrainTarget,TestTarget,1)

    def BinaryAfficheImageRandom(self,N):
        print("CLASS: ",end="")
        for ni in range(N):
            x = random.choice(range(self.NbTrain))
            print(" ",self.TrainTarget[x],end="")
            AfficheTextImage(self.TrainData[x], ni * 40 + 10,10)
        print()

class DataSet:
    def __init__(self, NameString, TrainData, TrainTarget, TestData, TestTarget):
        self.NameString = NameString                    # information on the data, name, etc.
        self.NbFeatures = len(TrainData[0])             # number of bit of each example
        self.NbObjectives = len(TrainTarget[0])            # number of targets (10 MNIST)
        self.NbTrain = len(TrainData)                   # number of training examples
        self.TrainData = numpy.asarray(TrainData)       # scikitlearning SVC train data
        self.TrainTarget = numpy.asarray(TrainTarget)   # scikitlearning SVC train target
        self.NbTest = len(TestData)                     # number of test examples
        self.TestData = numpy.asarray(TestData)         # scikitlearning SVC test data
        self.TestTarget = numpy.asarray(TestTarget)     # scikitlearning SVC test target
        
        self.Features = []                              # list of features for our algorithms (Train and
                                                        # test)
        self.Objectives = []                             # target of our algorithm

        TrainDataT = Transpose(TrainData)
        TestDataT = Transpose(TestData)
        TrainTargetT = Transpose(TrainTarget)
        TestTargetT = Transpose(TestTarget)

        self.Features = [  Feature(TrainDataT[i],  TestDataT[i],i) for i in range(self.NbFeatures)   ]
        self.Objectives = [ ObjectiveClass(TrainTargetT[i],TestTargetT[i],i) for i in range(self.NbObjectives) ] 

    def AfficheImageRandom(self,N):
        print("CLASS: ",end="",)
        for ni in range(N):
            x = random.choice(range(self.NbTrain))
            for i in range(len(self.TrainTarget[x])):
                if self.TrainTarget[x][i] == 1:
                    print(" ",i,end="")
                    break
                else:
                    continue
            AfficheTextImage(self.TrainData[x], ni * 40 + 10,10)
        print()

    #f = open(FileDump+".dump", "wb")
    #pickle.dump(Data,f)
    #f.close()
def LoadDataSet(File):
    f = open(File + ".dump","rb")
    Data = pickle.load(f)
    print("File loaded: " + File)
    print("File description: " + Data.NameString)
    print("Number of train example: ",Data.NbTrain)
    print("Number of test example: ",Data.NbTest)
    print()
    print()
    return Data

##############################################################################
##############################################################################
##############################################################################
# Convert a list of bit into its integer representation
def BitListToInt(bitlist):  #bitlist:list of bit
    out = 0
    for bit in bitlist:
        out = (out << 1) | bit
    return out

# convert the integer representation into a list of bit
def IntToBitList(n,NbBits):  #n:integer, NbBits:number of bit of the representationb (for pading with 0)
    list = [int(digit) for digit in bin(n)[2:]]
    return [0 for _ in range(NbBits - len(list)) ] + list    #list of bit

# return the integer representation of the bit by bit negation of the bit
# vector represented in Int.
def NegInt(Int,NbBits):    #int: number to negate, NbBits:total numberof bits for the representation.
    return Int ^ (2 ** NbBits - 1)


class Feature:
    def __init__(self, BitListTrain, BitListTest,FeatureNb):
        # initialyse from a list of 0 and 1
        self.FeatureNb = FeatureNb
        self.NbTrain = len(BitListTrain)
        self.NbTest = len(BitListTest)
        self.TrainValue = BitListToInt(BitListTrain)
        self.NegTrainValue = NegInt(self.TrainValue,self.NbTrain)
        self.TestValue = BitListToInt(BitListTest)
        self.NegTestValue = NegInt(self.TestValue,self.NbTest)

    def FeatureToListForTrain(self):
        #return the value of the feature as a list
        return IntToBitList(self.TrainValue,self.NbTrain)
    
    def FeatureToListForTest(self):
        return IntToBitList(self.TestValue, self.NbTest)

    def H(self):
        #Return the Shanon entropy of the TrainFeature
        return 0

def TPOfFeature(FeatureList):
    TrainList = [[x.TrainValue, x.NegTrainValue] for x in FeatureList] 
    TestList = [[x.TestValue, x.NegTestValue] for x in FeatureList] 
    # the product of a list of vector
    def TP(l1,l2):
        res = []
        for x in l1:
            for y in l2:
                res.append(x & y)
        return res
    return functools.reduce(TP,TrainList), functools.reduce(TP,TestList)

def vectorDistance(u,v):
    a = popcount(BitListToInt(u) ^ BitListToInt(v) )
    return a
    
def calculateNeighbors(arity):
    neighborsList = []
    lst = [list(i) for i in product([0, 1], repeat=arity)]
    lst.reverse()
    for i in lst:
        neighborListOfElement = []
        neighborsVect_d1 = []
        neighborsVect_d2 = []
        neighborsVect_d3 = []
        neighborsVect_d4 = []
        for j in lst:
            if i == j:
                continue
            else:
                if vectorDistance(i,j) == 1:
                    neighborsVect_d1.append(lst.index(j))
                elif vectorDistance(i,j) == 2:
                    neighborsVect_d2.append(lst.index(j))
                elif vectorDistance(i,j) == 3:
                    neighborsVect_d3.append(lst.index(j))
                elif vectorDistance(i,j) == 4:
                    neighborsVect_d4.append(lst.index(j))
        neighborListOfElement.append(neighborsVect_d1)
        neighborListOfElement.append(neighborsVect_d2)
        neighborListOfElement.append(neighborsVect_d3)
        neighborListOfElement.append(neighborsVect_d4)
        neighborsList.append(neighborListOfElement)
    return neighborsList

def combination(k,n):
    c = mth.factorial(n) / (mth.factorial(k) * mth.factorial(n-k))
    return c

def ErrorCalculate(hd,arity,Eps):
    if arity < 4:
        if hd > 2:
            sys.exit("Hamming distance must not exceed 2")
        else:
            d0 = (1-Eps) ** arity
            d1 = combination(1,arity) * Eps * (1 - Eps) ** (arity - 1)
            d2 = combination(2,arity) * Eps ** 2 * (1 - Eps) ** (arity - 2)
            if hd == 0:
                return [d0] 
            elif hd == 1:
                return [d0,d1]
            elif hd == 2:
                return [d0,d1,d2]
    else:
        d0 = (1-Eps) ** arity
        d1 = combination(1,arity) * Eps * (1 - Eps) ** (arity - 1)
        d2 = combination(2,arity) * Eps ** 2 * (1 - Eps) ** (arity - 2)
        d3 = combination(3,arity) * Eps ** 3 * (1 - Eps) ** (arity - 3)
        d4 = combination(4,arity) * Eps ** 4 * (1 - Eps) ** (arity - 4)
        if hd == 0:
            return [d0] 
        elif hd == 1:
            return [d0,d1]
        elif hd == 2:
            return [d0,d1,d2]
        elif hd == 3:
            return [d0,d1,d2,d3]
        elif hd == 4:
            return [d0,d1,d2,d3,d4]



class ObjectiveClass:
    def __init__(self, BitListTrain, BitListTest,NbObjective):
        # initialyse from a list of 0 and 1
        self.NbObjective = NbObjective
        self.NbTrain = len(BitListTrain) 
        self.NbTest = len(BitListTest)
        self.TrainOne = BitListToInt(BitListTrain)              #1 when the objective category has to be 1
        self.TrainZero = NegInt(self.TrainOne,self.NbTrain)     #1 when the objective category has to be 0
        self.TestOne = BitListToInt(BitListTest)                #1 when the objective category has to be 1
        self.TestZero = NegInt(self.TestOne,self.NbTest)        #1 when the objective category has to be 0

    def ObjectiveToListTrain(self):
        return IntToBitList(self.TrainOne,self.NbTrain)

    def ObjectiveToListTest(self):
        return IntToBitList(self.TestOne,self.NbTest)

def BinaryAccuracy1(Feature,Objective):
    TrainAcc_1 = popcount(Feature.TrainValue & Objective.TrainOne) / Feature.NbTrain
    TrainAcc_0 = popcount(Feature.NegTrainValue & Objective.TrainZero) / Feature.NbTrain
    TestAcc_1 = popcount(Feature.TestValue & Objective.TestOne) / Feature.NbTest
    TestAcc_0 = popcount(Feature.NegTestValue & Objective.TestZero) / Feature.NbTest
    return [[round(TrainAcc_1,4),round(TestAcc_1,4)],[round(TrainAcc_0,4),round(TestAcc_0,4)]]

def BaggingBinaryAccuracy1(Feature,Objective,mask):
    TrainAcc_1 = popcount(mask & Feature.TrainValue & Objective.TrainOne) / Feature.NbTrain
    TrainAcc_0 = popcount(mask & Feature.NegTrainValue & Objective.TrainZero) / Feature.NbTrain
    TestAcc_1 = popcount(Feature.TestValue & Objective.TestOne) / Feature.NbTest
    TestAcc_0 = popcount(Feature.NegTestValue & Objective.TestZero) / Feature.NbTest
    return [[round(TrainAcc_1,4),round(TestAcc_1,4)],[round(TrainAcc_0,4),round(TestAcc_0,4)]]

def BaggingBinaryAccuracy(Mask,Feature,Objective):
    #b = popcount(Mask)
    TrainAcc = (popcount(Mask & Feature.TrainValue & Objective.TrainZero) + popcount(Mask & Feature.NegTrainValue & Objective.TrainOne)) / popcount(Mask)
    TestAcc = (popcount(Feature.TestValue & Objective.TestZero) + popcount(Feature.NegTestValue & Objective.TestOne)) / Feature.NbTest
    return [round(1 - TrainAcc,4), round(1 - TestAcc,4)]

def BinaryAccuracy(Feature, Objective):
    TrainAcc = (popcount(Feature.TrainValue & Objective.TrainZero) + popcount(Feature.NegTrainValue & Objective.TrainOne)) / Feature.NbTrain
    TestAcc = (popcount(Feature.TestValue & Objective.TestZero) + popcount(Feature.NegTestValue & Objective.TestOne)) / Feature.NbTest
    return [round(1 - TrainAcc,4), round(1 - TestAcc,4)]

def BinaryPrintAccuracy(Feature,Objective):
    a = BinaryAccuracy(Feature,Objective)
   
    print("Train :",a[0])
    print("Test  :",a[1])
    print("Gap   :",round(a[0] - a[1],4))
    print("-----------------")
    pt = graphics.Point(posx[0], 600 - (a[0] * 1000 - 500))
    pt.setFill("red")
    pt.draw(win)
    pt = graphics.Point(posx[0], 600 - (a[1] * 1000 - 500))
    pt.setFill("blue")
    pt.draw(win)
    posx[0] = posx[0] + 1

def Accuracy(FeatureListTrain,FeatureListTest,ObjectiveListTrain,ObjectiveListTest):
    TrainAcc = 0
    TestAcc = 0
    ones = BitListToInt([1 for i in range(10)])
    for i in range(len(FeatureListTrain)):
        if FeatureListTrain[i] == 0:
            TrainAcc = TrainAcc + 1/10
        elif popcount(FeatureListTrain[i] & ObjectiveListTrain[i]) == 0: 
            TrainAcc = TrainAcc + 0
        else:
            if popcount(ones ^ (FeatureListTrain[i] ^ ObjectiveListTrain[i])) == 10:
                TrainAcc = TrainAcc + 1
            else:
                TrainAcc = TrainAcc + 1/popcount(FeatureListTrain[i])
    TrainAcc = TrainAcc/ len(FeatureListTrain)

    for j in range(len(FeatureListTest)):
        if FeatureListTest[j] == 0:
            TestAcc = TestAcc + 1/10
        elif popcount(FeatureListTest[j] & ObjectiveListTest[j]) == 0:
            TestAcc = TestAcc + 0
        else:
            if popcount(ones ^ (FeatureListTest[j] ^ ObjectiveListTest[j])) == 10:
                TestAcc = TestAcc + 1
            else:
                TestAcc = TestAcc + 1/popcount(FeatureListTest[j])
    TestAcc = TestAcc/len(FeatureListTest)
    return [round(TrainAcc,4), round(TestAcc,4)]
    

def PrintAccuracy(FeatureListTrain,FeatureListTest,ObjectiveListTrain,ObjectiveListTest):
    a = Accuracy(FeatureListTrain,FeatureListTest,ObjectiveListTrain,ObjectiveListTest)
   
    print("Train :",a[0])
    print("Test  :",a[1])
    print("Gap   :",round(a[0] - a[1],4))
    print("-----------------")
    pt = graphics.Point(posx[0], 600 - (a[0] * 1000 - 200))
    pt.setFill("red")
    pt.draw(win)
    pt = graphics.Point(posx[0], 600 - (a[1] * 1000 - 200))
    pt.setFill("blue")
    pt.draw(win)
    #pt = graphics.Point(posx[0], 800 - (a[2] * 1000))
    #pt.setFill("green")
    #pt.draw(win)
    posx[0] = posx[0] + 1




class Node:
    # initialyse from a feature vector
    def __init__(self, Arity, Objective, Parent):
        self.Arity = Arity
        self.Feature = Feature([0],[0],0)
        self.Childrens = []
        self.Objective = copy.copy(Objective)
        self.Parent = Parent
        self.TruthTable = []
        self.IsLeaf = False
        self.IsHead = False
        self.IsRoot = False
        self.FeatureList = []
    
    #give the level in the graph of a node
    def getLevel(self):
        level = 1
        if self.IsHead:
            return level
        #else:
        #    return 0
        else: 
            parent = self.ParentsWithoutRepetition()[0]
            while parent.IsHead == False:
                level = level + 1
                parent = parent.ParentsWithoutRepetition()[0]
            level = level + 1
            return level

    #calculate a feature according to a truthtable (gate)
    def EvaluateTruthTable(self):
              
        if self.IsLeaf:
            cf = self.Childrens
        else:
            cf = [x.Feature for x in self.Childrens]
        self.Feature.NbTrain = cf[0].NbTrain
        self.Feature.NbTest = cf[0].NbTest
        tpTrain, tpTest = TPOfFeature(cf)
        
        self.Feature.TrainValue = 0
        self.Feature.NegTrainValue = 0
        self.Feature.TestValue = 0
        self.Feature.NegTestValue = 0

        for i in range(len(tpTrain)):
            if self.TruthTable[i] == 1:
                self.Feature.TrainValue = self.Feature.TrainValue ^ tpTrain[i]
                self.Feature.TestValue = self.Feature.TestValue ^ tpTest[i]
            else:
                self.Feature.NegTrainValue = self.Feature.NegTrainValue ^ tpTrain[i]
                self.Feature.NegTestValue = self.Feature.NegTestValue ^ tpTest[i]

    ########### version information

    def OptimizeTruthTable(self,Levels,hd,Eps_top,Eps_bottom,neighborsList):
        
        N = self.Objective.NbTrain
        
        if self.IsLeaf:
            cf = self.Childrens
        else:
            cf = [x.Feature for x in self.Childrens]
        self.Feature.NbTrain = cf[0].NbTrain
        self.Feature.NbTest = cf[0].NbTest
        tpTrain, tpTest = TPOfFeature(cf)

        if self.IsHead:
            list_d = ErrorCalculate(hd,self.Arity,Eps_top)
        elif self.getLevel() == len(Levels):
            list_d = ErrorCalculate(hd,self.Arity,Eps_bottom)
        else:
            level = self.getLevel()
            w = len(Levels)
            Eps_interpolated = ((w - level) / ( w - 1)) * Eps_top + ((level - 1) / (w - 1)) * Eps_bottom
            list_d = ErrorCalculate(hd,self.Arity,Eps_interpolated)

        l=len(tpTrain)
        l1 = []
        l2 = []
        for i in range(l):
            g1 = popcount(tpTrain[i] & self.Objective.TrainOne) * list_d[0]
            g0 = popcount(tpTrain[i] & self.Objective.TrainZero) * list_d[0]
            if len(list_d) > 1:
                for j in neighborsList[i][0]:
                    g1 = g1 + (popcount(tpTrain[j] & self.Objective.TrainOne) * list_d[1])/len(neighborsList[0][0])
                    g0 = g0 + (popcount(tpTrain[j] & self.Objective.TrainZero) * list_d[1])/len(neighborsList[0][0])
            if len(list_d) > 2:
                for k in neighborsList[i][1]:
                    g1 = g1 + (popcount(tpTrain[k] & self.Objective.TrainOne) * list_d[2])/len(neighborsList[0][1])
                    g0 = g0 + (popcount(tpTrain[k] & self.Objective.TrainZero) * list_d[2])/len(neighborsList[0][1])
            if len(list_d) > 3:
                for m in neighborsList[i][2]:
                    g1 = g1 + (popcount(tpTrain[m] & self.Objective.TrainOne) * list_d[3])/len(neighborsList[0][2])
                    g0 = g0 + (popcount(tpTrain[m] & self.Objective.TrainZero) * list_d[3])/len(neighborsList[0][2])
            if len(list_d) > 4:
                for n in neighborsList[i][3]:
                    g1 = g1 + (popcount(tpTrain[n] & self.Objective.TrainOne) * list_d[4])/len(neighborsList[0][3])
                    g0 = g0 + (popcount(tpTrain[n] & self.Objective.TrainZero) * list_d[4])/len(neighborsList[0][3])
            l1.append(g1)
            l2.append(g0)

        l3 = [[l1[i] /(l1[i] + l2[i]+1), l1[i], l2[i],i] for i in range(l)]
        l3 = Transpose(reversed(sorted(l3)))
        

        tot1 = sum(l3[1])
        lt = []
        v1 = 0
        v2 = 0
        for i in range(0,l):
            v1 = v1 + l3[1][i]
            v2 = v2 + l3[2][i]
            tot = max(v1 + v2, 1)
            tot = min(tot,N - 1) #s'il y a que cette combinaison du pd tensoriel alors prendre tot à N-1
            val = (1 - ((tot / N) * H(v1 / tot) + (N - tot) / N * H((tot1 - v1) / (N - tot)))) # pourquoi tu divises sur log2 dans ta formule d'entropie
            lt.append(val)
             
        pm = PosMax(lt)

        self.Feature.TrainValue = 0
        self.Feature.NegTrainValue = 0
        self.Feature.TestValue = 0
        self.Feature.NegTestValue = 0

        s0 = 0
        sol = [0 for _ in range(l)]
        self.TruthTable=[0 for _ in range(l)]
        for i in range(l):
            pos = l3[3][i]
            x = tpTrain[pos]
            x2 = tpTest[pos]
            if i <= pm:
                self.Feature.TrainValue = self.Feature.TrainValue ^ x
                self.Feature.TestValue = self.Feature.TestValue ^ x2
                self.TruthTable[pos]=1  
            else:
                self.Feature.NegTrainValue = self.Feature.NegTrainValue ^ x
                self.Feature.NegTestValue = self.Feature.NegTestValue ^ x2
                self.TruthTable[pos]=0    
        #self.EvaluateTruthTable()  
        
    def BaggingOptimizeTruthTable(self,Levels,hd,Eps_top,Eps_bottom,neighborsList,mask):

        N = popcount(mask)
        
        if self.IsLeaf:
            cf = self.Childrens
        else:
            cf = [x.Feature for x in self.Childrens]
        self.Feature.NbTrain = cf[0].NbTrain
        self.Feature.NbTest = cf[0].NbTest
        tpTrain, tpTest = TPOfFeature(cf)

        if self.IsHead:
            list_d = ErrorCalculate(hd,self.Arity,Eps_top)
        elif self.getLevel() == len(Levels):
            list_d = ErrorCalculate(hd,self.Arity,Eps_bottom)
        else:
            level = self.getLevel()
            w = len(Levels)
            Eps_interpolated = ((w - level) / ( w - 1)) * Eps_top + ((level - 1) / (w - 1)) * Eps_bottom
            list_d = ErrorCalculate(hd,self.Arity,Eps_interpolated)

        l=len(tpTrain)
        l1 = []
        l2 = []
        for i in range(l):
            g1 = popcount(mask & tpTrain[i] & self.Objective.TrainOne) * list_d[0]
            g0 = popcount(mask & tpTrain[i] & self.Objective.TrainZero) * list_d[0]
            if len(list_d) > 1:
                for j in neighborsList[i][0]:
                    g1 = g1 + (popcount(mask & tpTrain[j] & self.Objective.TrainOne) * list_d[1])/len(neighborsList[0][0])
                    g0 = g0 + (popcount(mask & tpTrain[j] & self.Objective.TrainZero) * list_d[1])/len(neighborsList[0][0])
            if len(list_d) > 2:
                for k in neighborsList[i][1]:
                    g1 = g1 + (popcount(mask & tpTrain[k] & self.Objective.TrainOne) * list_d[2])/len(neighborsList[0][1])
                    g0 = g0 + (popcount(mask & tpTrain[k] & self.Objective.TrainZero) * list_d[2])/len(neighborsList[0][1])
            if len(list_d) > 3:
                for m in neighborsList[i][2]:
                    g1 = g1 + (popcount(mask & tpTrain[m] & self.Objective.TrainOne) * list_d[3])/len(neighborsList[0][2])
                    g0 = g0 + (popcount(mask & tpTrain[m] & self.Objective.TrainZero) * list_d[3])/len(neighborsList[0][2])
            if len(list_d) > 4:
                for n in neighborsList[i][3]:
                    g1 = g1 + (popcount(mask & tpTrain[n] & self.Objective.TrainOne) * list_d[4])/len(neighborsList[0][3])
                    g0 = g0 + (popcount(mask & tpTrain[n] & self.Objective.TrainZero) * list_d[4])/len(neighborsList[0][3])
            l1.append(g1)
            l2.append(g0)

        l3 = [[l1[i] /(l1[i] + l2[i]+1), l1[i], l2[i],i] for i in range(l)]
        l3 = Transpose(reversed(sorted(l3)))
        

        tot1 = sum(l3[1])
        lt = []
        v1 = 0
        v2 = 0
        for i in range(0,l):
            v1 = v1 + l3[1][i]
            v2 = v2 + l3[2][i]
            tot = max(v1 + v2, 1)
            tot = min(tot,N - 1) #s'il y a que cette combinaison du pd tensoriel alors prendre tot à N-1
            val = (1 - ((tot / N) * H(v1 / tot) + (N - tot) / N * H((tot1 - v1) / (N - tot)))) # pourquoi tu divises sur log2 dans ta formule d'entropie
            lt.append(val)
             
        pm = PosMax(lt)

        self.Feature.TrainValue = 0
        self.Feature.NegTrainValue = 0
        self.Feature.TestValue = 0
        self.Feature.NegTestValue = 0

        s0 = 0
        sol = [0 for _ in range(l)]
        self.TruthTable=[0 for _ in range(l)]
        for i in range(l):
            pos = l3[3][i]
            x = tpTrain[pos]
            x2 = tpTest[pos]
            if i <= pm:
                self.Feature.TrainValue = self.Feature.TrainValue ^ x
                self.Feature.TestValue = self.Feature.TestValue ^ x2
                self.TruthTable[pos]=1  
            else:
                self.Feature.NegTrainValue = self.Feature.NegTrainValue ^ x
                self.Feature.NegTestValue = self.Feature.NegTestValue ^ x2
                self.TruthTable[pos]=0
         

    def NaiveOptimizeTruthTable(self):

        if self.IsLeaf:
            cf = self.Childrens
        else:
            cf = [x.Feature for x in self.Childrens]
        tpTrain, tpTest = TPOfFeature(cf)
        l = len(tpTrain)
        for i in range(l):
            self.TruthTable.append(random.randint(0,1))

        self.EvaluateTruthTable()

   
    def OptimizeTruthTableProb(self,Levels,hd,Eps_top,Eps_bottom,neighborsList):
        ########### version prob
        
        if self.IsLeaf:
            cf = self.Childrens
        else:
            cf = [x.Feature for x in self.Childrens]
        self.Feature.NbTrain = cf[0].NbTrain
        self.Feature.NbTest = cf[0].NbTest
        tpTrain, tpTest = TPOfFeature(cf)
        #neighborsList = calculateNeighbors(self.Arity)

        if self.IsHead:
            list_d = ErrorCalculate(hd,self.Arity,Eps_top)
        elif self.getLevel() == len(Levels):
            list_d = ErrorCalculate(hd,self.Arity,Eps_bottom)
        else:
            level = self.getLevel()
            w = len(Levels)
            Eps_interpolated = ((w - level) / ( w - 1)) * Eps_top + ((level - 1) / (w - 1)) * Eps_bottom
            list_d = ErrorCalculate(hd,self.Arity,Eps_interpolated)

        self.Feature.TrainValue = 0
        self.Feature.NegTrainValue = 0
        self.Feature.TestValue = 0
        self.Feature.NegTestValue = 0

        self.TruthTable = []
        for i in range(len(tpTrain)):
            e = 0.0
            g1 = popcount(tpTrain[i] & self.Objective.TrainOne) * list_d[0]
            g0 = popcount(tpTrain[i] & self.Objective.TrainZero) * list_d[0]
            if len(list_d) > 1:
                for j in neighborsList[i][0]:
                    g1 = g1 + (popcount(tpTrain[j] & self.Objective.TrainOne) * list_d[1])/len(neighborsList[0][0])
                    g0 = g0 + (popcount(tpTrain[j] & self.Objective.TrainZero) * list_d[1])/len(neighborsList[0][0])
            if len(list_d) > 2:
                for k in neighborsList[i][1]:
                    g1 = g1 + (popcount(tpTrain[k] & self.Objective.TrainOne) * list_d[2])/len(neighborsList[0][1])
                    g0 = g0 + (popcount(tpTrain[k] & self.Objective.TrainZero) * list_d[2])/len(neighborsList[0][1])
            if len(list_d) > 3:
                for m in neighborsList[i][2]:
                    g1 = g1 + (popcount(tpTrain[m] & self.Objective.TrainOne) * list_d[3])/len(neighborsList[0][2])
                    g0 = g0 + (popcount(tpTrain[m] & self.Objective.TrainZero) * list_d[3])/len(neighborsList[0][2])
            if len(list_d) > 4:
                for n in neighborsList[i][3]:
                    g1 = g1 + (popcount(tpTrain[n] & self.Objective.TrainOne) * list_d[4])/len(neighborsList[0][3])
                    g0 = g0 + (popcount(tpTrain[n] & self.Objective.TrainZero) * list_d[4])/len(neighborsList[0][3])
            e = g1 - g0
            if (e > 0.0):
                self.Feature.TrainValue = self.Feature.TrainValue ^ tpTrain[i]
                self.Feature.TestValue = self.Feature.TestValue ^ tpTest[i]
                self.TruthTable.append(1)
            else:
                self.Feature.NegTrainValue = self.Feature.NegTrainValue ^ tpTrain[i]
                self.Feature.NegTestValue = self.Feature.NegTestValue ^ tpTest[i]
                self.TruthTable.append(0)

    def BinaryOptimizeTruthTableSubGraph(self,Levels,hd,Eps_top,Eps_bottom,neighborsList,choice,mask):
        if self.IsLeaf == False:
            for x in self.ChildrenWithoutRepetition():
                if x.TruthTable == []:
                    x.BinaryOptimizeTruthTableSubGraph(Levels,hd,Eps_top,Eps_bottom,neighborsList,choice,mask)
        if self.TruthTable == []:
            if mask != None:
                self.BaggingOptimizeTruthTable(Levels,hd,Eps_top,Eps_bottom,neighborsList,mask)
            else:
                if choice == 0:
                    self.OptimizeTruthTable(Levels,hd,Eps_top,Eps_bottom,neighborsList)
                else:
                    self.OptimizeTruthTableProb(Levels,hd,Eps_top,Eps_bottom,neighborsList)

    def OptimizeTruthTableSubGraph(self,Levels,hd,Eps_top,Eps_bottom,neighborsList):
        if self.IsLeaf == False:
            for x in self.ChildrenWithoutRepetition():
                if x.TruthTable == []:
                    x.OptimizeTruthTableSubGraph(Levels,hd,Eps_top,Eps_bottom,neighborsList)
        if self.TruthTable == []:
            if self.IsRoot == False:
                self.OptimizeTruthTable(Levels,hd,Eps_top,Eps_bottom,neighborsList)
            else:
                self.FeatureList = [i.Feature for i in self.Childrens]

    def ParentsWithoutRepetition(self):
        if isinstance(self.Parent,Node):
            lst = []
            lst.append(self.Parent)
            return lst
        else:
            return list(set(self.Parent))

    def ChildrenWithoutRepetition(self):
        return list(set(self.Childrens))
    
    def RandomLeaf(self):
        if self.IsLeaf:
            return self
        else:
            ch = random.choice(self.Childrens)
            return ch.RandomLeaf()

    def RandomChild(self,Depth):
        if self.IsLeaf or Depth == 1:
            return self
        else:
            ch = random.choice(self.Childrens)
            return ch.RandomChild(Depth - 1)
    
    def OptimizeTruthTableToTop(self,Levels,hd,Eps_top,Eps_bottom,neighborsList,choice,mask):
        if choice == 0:
            if mask == None:
                self.OptimizeTruthTable(Levels,hd,Eps_top,Eps_bottom,neighborsList)
            else:
                self.BaggingOptimizeTruthTable(Levels,hd,Eps_top,Eps_bottom,neighborsList,mask)
        else:
            self.OptimizeTruthTableProb(Levels,hd,Eps_top,Eps_bottom,neighborsList)
        self.ToTopTruthTable(Levels,hd,Eps_top,Eps_bottom,neighborsList,choice,mask)

    def ToTopTruthTable(self,Levels,hd,Eps_top,Eps_bottom,neighborsList,choice,mask):
        if isinstance(self.Parent,list):
            for i in self.ParentsWithoutRepetition():
                if choice == 0:
                    if mask == None:
                        i.OptimizeTruthTable(Levels,hd,Eps_top,Eps_bottom,neighborsList)
                    else:
                        i.BaggingOptimizeTruthTable(Levels,hd,Eps_top,Eps_bottom,neighborsList,mask)
                else:
                    i.OptimizeTruthTableProb(Levels,hd,Eps_top,Eps_bottom,neighborsList)
            Parents = RetrieveNodeParents(self.Parent)
            c = 0
            x = False
            while c >= 0 and x == False:
                if len(Parents) == 1 and Parents[0].IsHead == True:
                    if choice == 0:
                        if mask == None:
                            Parents[0].OptimizeTruthTable(Levels,hd,Eps_top,Eps_bottom,neighborsList)
                        else:
                            Parents[0].BaggingOptimizeTruthTable(Levels,hd,Eps_top,Eps_bottom,neighborsList,mask)
                        x = True
                        c = c + 1
                    else:
                        Parents[0].OptimizeTruthTableProb(Levels,hd,Eps_top,Eps_bottom,neighborsList)
                        x = True
                        c = c + 1
                else:
                    for j in Parents:
                        if choice == 0:
                            if mask == None:
                                j.OptimizeTruthTable(Levels,hd,Eps_top,Eps_bottom,neighborsList)
                            else:
                                j.BaggingOptimizeTruthTable(Levels,hd,Eps_top,Eps_bottom,neighborsList,mask)
                        else:
                            j.OptimizeTruthTableProb(Levels,hd,Eps_top,Eps_bottom,neighborsList)
                    Parents = RetrieveNodeParents(Parents)
                    c = c + 1
        else:
            if choice == 0:
                if mask == 0:
                    self.Parent.OptimizeTruthTable(Levels,hd,Eps_top,Eps_bottom,neighborsList)
                else:
                    self.Parent.BaggingOptimizeTruthTable(Levels,hd,Eps_top,Eps_bottom,neighborsList,mask)
            else:
                self.Parent.OptimizeTruthTableProb(Levels,hd,Eps_top,Eps_bottom,neighborsList)

    def OptimizeTruthTableToTopProb(self,Levels,hd,Eps_top,Eps_bottom,neighborsList):
        self.OptimizeTruthTableProb(Levels,hd,Eps_top,Eps_bottom,neighborsList)
        if self.IsHead == False:
            self.Parent.OptimizeTruthTableToTopProb(Levels,hd,Eps_top,Eps_bottom,neighborsList)

    def TopToHead(self):
        node = copy.copy(self)
        parents = node.ParentsWithoutRepetition()
        while parents[0].IsRoot == False:
            if parents[0].IsHead == True:
                return parents
            else:
                parents = RetrieveNodeParents(parents)

    def ToTop(self):
        for i in self.ParentsWithoutRepetition():
            i.EvaluateTruthTable()
        Parents = RetrieveNodeParents(self.ParentsWithoutRepetition())
       
        while Parents[0].IsRoot == False :
            for j in Parents:
                j.EvaluateTruthTable()
            Parents = RetrieveNodeParents(Parents)


    def BinaryTopToHead(self):
        node = copy.copy(self)
        while node.IsHead == False:
            if isinstance(node.Parent,list):
                node = node.Parent[0]
            else:
                node = node.Parent
        return node 

    def BinaryToTop(self):
        for i in self.ParentsWithoutRepetition():
            i.EvaluateTruthTable()
        if self.ParentsWithoutRepetition()[0].IsHead == True:
            self.Parent.EvaluateTruthTable()
        else:
            Parents = RetrieveNodeParents(self.ParentsWithoutRepetition())
       
            while Parents[0].IsHead == False :
                for j in Parents:
                    j.EvaluateTruthTable()
                Parents = RetrieveNodeParents(Parents)
            Parents[0].EvaluateTruthTable()

    def BinaryOptimizeFeatureFromTop(self,hd,list_d,neighborsList,mask):

        if self.IsLeaf:
            cf = self.Childrens
        else:
            cf = [x.Feature for x in self.Childrens]
        tpTrain, tpTest = TPOfFeature(cf)
        
        Featurebakup = copy.copy(self.Feature)

        one = Featurebakup.TrainValue ^ Featurebakup.NegTrainValue

        self.Feature.TrainValue = one 
        self.Feature.NegTrainValue = 0
        if self.IsHead == False:
            self.BinaryToTop()
            if mask == None:
                z1 = self.BinaryTopToHead().Feature.TrainValue
            else:
                z1 = self.BinaryTopToHead().Feature.TrainValue & mask
        else:
            if mask == None:
                z1 = self.Feature.TrainValue
            else:
                z1 = self.Feature.TrainValue & mask

        self.Feature.TrainValue = 0 
        self.Feature.NegTrainValue = one
        if self.IsHead == False:
            self.BinaryToTop()
            if mask == None:
                z0 = self.BinaryTopToHead().Feature.TrainValue
            else:
                z0 = self.BinaryTopToHead().Feature.TrainValue & mask
        else:
            if mask == None:
                z0 = self.Feature.TrainValue
            else:
                z0 = self.Feature.TrainValue & mask

        for i in range(len(tpTrain)):
            count1_one = popcount((one ^ (z1 ^ self.Objective.TrainOne)) & tpTrain[i]) * list_d[0]
            count1_zero = popcount((one ^ (z0 ^ self.Objective.TrainOne)) & tpTrain[i]) * list_d[0]
            if len(list_d) > 1:
                for j in neighborsList[i][0]:
                    count1_one = count1_one + (popcount((one ^ (z1 ^ self.Objective.TrainOne)) & tpTrain[j]) * list_d[1]) / len(neighborsList[0][0])
                    count1_zero = count1_zero + (popcount((one ^ (z0 ^ self.Objective.TrainOne)) & tpTrain[j]) * list_d[1]) / len(neighborsList[0][0])
            if len(list_d) > 2:
                for k in neighborsList[i][1]:
                    count1_one = count1_one + (popcount((one ^ (z1 ^ self.Objective.TrainOne)) & tpTrain[k]) * list_d[2]) / len(neighborsList[0][1])
                    count1_zero = count1_zero + (popcount((one ^ (z0 ^ self.Objective.TrainOne)) & tpTrain[k]) * list_d[2]) / len(neighborsList[0][1])
            if len(list_d) > 3:
                for h in neighborsList[i][2]:
                    count1_one = count1_one + (popcount((one ^ (z1 ^ self.Objective.TrainOne)) & tpTrain[h]) * list_d[3]) / len(neighborsList[0][2])
                    count1_zero = count1_zero + (popcount((one ^ (z0 ^ self.Objective.TrainOne)) & tpTrain[h]) * list_d[3]) / len(neighborsList[0][2])
            if len(list_d) > 4:
                for g in neighborsList[i][3]:
                    count1_one = count1_one + (popcount((one ^ (z1 ^ self.Objective.TrainOne)) & tpTrain[g]) * list_d[4]) / len(neighborsList[0][3])
                    count1_zero = count1_zero + (popcount((one ^ (z0 ^ self.Objective.TrainOne)) & tpTrain[g]) * list_d[4]) / len(neighborsList[0][3])
            if count1_one - count1_zero >= 0:
                self.TruthTable[i] = 1
            else:
                self.TruthTable[i] = 0
        
        if self.IsHead == False:
            self.EvaluateTruthTable()
            self.BinaryToTop()
        else:
            self.EvaluateTruthTable()

        
    def OptimizeFeatureFromTop(self,hd,list_d,neighborsList,c):

        if self.IsLeaf:
            cf = self.Childrens
        else:
            cf = [x.Feature for x in self.Childrens]
        tpTrain, tpTest = TPOfFeature(cf)
        
        Featurebakup = copy.copy(self.Feature)

        one = Featurebakup.TrainValue ^ Featurebakup.NegTrainValue

        #l = []
        #if self.IsHead == True:
        #    nodes = [self]
        #else:
        #    nodes = self.TopToHead()
        #for i in range(len(tpTrain)):
        #    sum = 0
        #    for j in nodes:
        #        sum = sum + popcount((j.Feature.TrainValue & j.Objective.TrainOne) & tpTrain[i]) + popcount((j.Feature.NegTrainValue & j.Objective.TrainZero) & tpTrain[i])
        #    l.append(sum)


        self.Feature.TrainValue = one 
        self.Feature.NegTrainValue = 0
        
        if self.IsHead == True:
            headFeaturesOf1 = [self]
        else:
            self.ToTop()
            headFeaturesOf1 = self.TopToHead()

        lst1 = []
        for i in range(len(tpTrain)):
            sum_one = 0
            for x in headFeaturesOf1:
                sum_one = sum_one + popcount((x.Feature.TrainValue & x.Objective.TrainOne) & tpTrain[i]) * c * list_d[0] + popcount((x.Feature.NegTrainValue & x.Objective.TrainZero) & tpTrain[i]) * list_d[0]
                if len(list_d)> 1:
                    for j in neighborsList[i][0]:
                        sum_one = sum_one + ( popcount((x.Feature.TrainValue & x.Objective.TrainOne) & tpTrain[j]) * c * list_d[1]
                                + popcount((x.Feature.NegTrainValue & x.Objective.TrainZero) & tpTrain[j] ) * list_d[1] ) / len(neighborsList[0][0])
                    if len(list_d) > 2:
                        for k in neighborsList[i][1]:
                            sum_one = sum_one + ( popcount((x.Feature.TrainValue & x.Objective.TrainOne) & tpTrain[k]) * c * list_d[2] 
                                    + popcount((x.Feature.NegTrainValue & x.Objective.TrainZero) & tpTrain[k] ) * list_d[2] ) / len(neighborsList[0][1])
                        if len(list_d) > 3:
                            for h in neighborsList[i][2]:
                                sum_one = sum_one + ( popcount((x.Feature.TrainValue & x.Objective.TrainOne) & tpTrain[h]) * c * list_d[3]
                                    + popcount((x.Feature.NegTrainValue & x.Objective.TrainZero) & tpTrain[h] ) * list_d[3] ) / len(neighborsList[0][2])
                            if len(list_d) > 4:
                                for g in neighborsList[i][3]:
                                    sum_one = sum_one + ( popcount((x.Feature.TrainValue & x.Objective.TrainOne) & tpTrain[g]) * c * list_d[4]
                                        + popcount((x.Feature.NegTrainValue & x.Objective.TrainZero) & tpTrain[g] ) * list_d[4] ) / len(neighborsList[0][3])
            lst1.append(sum_one)


        self.Feature.TrainValue = 0 
        self.Feature.NegTrainValue = one
        
        if self.IsHead == True:
            headFeaturesOf0 = [self]
        else:
            self.ToTop()
            headFeaturesOf0 = self.TopToHead()

        lst0 = []
        for v in range(len(tpTrain)):
            sum_zero = 0
            for w in headFeaturesOf0:
                sum_zero = sum_zero + popcount((w.Feature.TrainValue & w.Objective.TrainOne) & tpTrain[v]) * c * list_d[0] + popcount((w.Feature.NegTrainValue & w.Objective.TrainZero) & tpTrain[v]) * list_d[0]
                if len(list_d) > 1:
                    for j in neighborsList[v][0]:
                        sum_zero = sum_zero + ( popcount((w.Feature.TrainValue & w.Objective.TrainOne) & tpTrain[j]) * c * list_d[1] 
                           + popcount((w.Feature.NegTrainValue & w.Objective.TrainZero) & tpTrain[j] ) * list_d[1] ) / len(neighborsList[0][0])
                    if len(list_d) > 2:
                        for k in neighborsList[v][1]:
                            sum_zero = sum_zero + ( popcount((w.Feature.TrainValue & w.Objective.TrainOne) & tpTrain[k]) * c * list_d[2] 
                               + popcount((w.Feature.NegTrainValue & w.Objective.TrainZero) & tpTrain[k] ) * list_d[2] ) / len(neighborsList[0][1])
                        if len(list_d) > 3:
                            for h in neighborsList[v][2]:
                                sum_zero = sum_zero + ( popcount((w.Feature.TrainValue & w.Objective.TrainOne) & tpTrain[h]) * c * list_d[3]
                                    + popcount((w.Feature.NegTrainValue & w.Objective.TrainZero) & tpTrain[h] ) * list_d[3] ) / len(neighborsList[0][2])
                            if len(list_d) > 4:
                                for g in neighborsList[v][3]:
                                    sum_zero = sum_zero + ( popcount((w.Feature.TrainValue & w.Objective.TrainOne) & tpTrain[g]) * c * list_d[4]
                                        + popcount((w.Feature.NegTrainValue & w.Objective.TrainZero) & tpTrain[g] ) * list_d[4] ) / len(neighborsList[0][3])

            lst0.append(sum_zero)

        for p in range(len(tpTrain)):
            if lst1[p] - lst0[p] >= 0:
                self.TruthTable[p] = 1
            else:
                self.TruthTable[p] = 0

        self.EvaluateTruthTable()
        if self.IsHead == False:
            self.ToTop()

        #l1 = []
        #if self.IsHead == True:
        #    nodes1 = [self]
        #else:
        #    nodes1 = self.TopToHead()
        #for i in range(len(tpTrain)):
        #    sum = 0
        #    for j in nodes:
        #        sum = sum + popcount((j.Feature.TrainValue & j.Objective.TrainOne) & tpTrain[i]) + popcount((j.Feature.NegTrainValue & j.Objective.TrainZero) & tpTrain[i])
        #    l1.append(sum)

        #print(l)
        #print(l1)
        #print("mouna")

               
def RetrieveNodeParents(NodeList):
    ParentList = []
    for i in NodeList:
        if isinstance(i.Parent,list):
            for j in i.Parent:
                c = j in ParentList
                if c == False:
                    ParentList.append(j)
        else:
            b = i.Parent in ParentList
            if b == False:
                ParentList.append(i.Parent)
    return ParentList

def RetrieveNodeChildren(NodeList):
    ChildrenList = []
    for i in NodeList:
        for j in i.Childrens:
            c = j in ChildrenList
            if c == False:
                ChildrenList.append(j)
    return ChildrenList
        
def ListOfNumberGenerator(number,nrepeat):
    list = [i for _ in range(nrepeat) for i in range(number)]
    shuffle(list)
    return list

def ListOfNumberGeneratorWithoutRepetition(number,nrepeat):
    i = 0
    list = []
    l = [i for i in range(number)]
    shuffle(l)
    list.extend(l)
    while i < (nrepeat-1):
        shuffle(l)
        if list[len(list) - 1] != l[0]:
            list.extend(l)
            i = i + 1
        else:
            shuffle(l)
    return list

def BinaryListOfNodesGenerator(number,Arity,Objective):
    list = [Node(Arity,Objective,[]) for _ in range(number)]
    return list

def ListOfNodesGenerator(number,Arity,ObjectiveList,Target_sets_value):
    list = []
    for _ in range(number):
        if Target_sets_value == True:
            c = random.randint(1,len(ObjectiveList)-1)
            targets = random.sample(ObjectiveList,c)
            new_target = OrBetweenObjectives(targets)
            list.append(Node(Arity,new_target,[]))
        else:
            k = random.randint(0,len(ObjectiveList)-1)
            list.append(Node(Arity,ObjectiveList[k],[]))
    return list

def VerifyLevels(list,Arity,ObjectiveList):
    if list[1] == len(ObjectiveList):
        for i in range(2,len(list) - 1):
            if (list[i]*Arity) % list[i+1] != 0:
                print("Change the levels values")
                return False
        return True
    else:
        print("the number of nodes of the second level has to be equal to the number of classes")
        return False

def BinaryVerifyLevels(list,Arity):
    if list[1] == Arity:
        for i in range(2,len(list) - 1):
            if (list[i]*Arity) % list[i+1] != 0:
                print("Change the levels values")
                return False
        return True
    else:
        print("the number of nodes of the second level has to be equal to the arity")
        return False

def BinaryCreateEmptyGraph(Arity, Levels, Objective, FeatureList, Parent):
    if BinaryVerifyLevels(Levels,Arity) == False:
        sys.exit("Verify Levels of the Graph")
    else:
        node = Node(Arity,Objective,Parent)
        node.Childrens = [Node(Arity,Objective,node) for _ in range(Levels[1])]
        level_parents = node.Childrens
        for i in range(2,len(Levels)):
            ListOfNumbers = ListOfNumberGeneratorWithoutRepetition(Levels[i],int((Levels[i-1]*Arity)/Levels[i]))
            ListOfNodes = BinaryListOfNodesGenerator(Levels[i],Arity,Objective)
            count = 0
            for j in level_parents:
                for k in range(Arity):
                    ListOfNodes[ListOfNumbers[count*Arity+ k]].Parent.append(j)
                    j.Childrens.append(ListOfNodes[ListOfNumbers[count*Arity+k]])
                count = count + 1
            level_parents = ListOfNodes
        for c in level_parents:
            c.Childrens = random.sample(FeatureList,Arity)
            c.IsLeaf = True
        return node
        
def BinaryCreateEmptyGraphAllFeature(Arity, Levels, Objective, FeatureList, Parent, LeafNo):
    if BinaryVerifyLevels(Levels,Arity) == False:
        sys.exit("Verify Levels of the Graph")
    else:

        node = Node(Arity,Objective,Parent)
        node.Childrens = [Node(Arity,Objective,node) for _ in range(Levels[1])]
        level_parents = node.Childrens
        for i in range(2,len(Levels)):
            ListOfNumbers = ListOfNumberGeneratorWithoutRepetition(Levels[i],(Levels[i-1]*Arity)/Levels[i])
            ListOfNodes = BinaryListOfNodesGenerator(Levels[i],Arity,Objective)
            count = 0
            for j in level_parents:
                for k in range(Arity):
                    ListOfNodes[ListOfNumbers[count*Arity+ k]].Parent.append(j)
                    j.Childrens.append(ListOfNodes[ListOfNumbers[count*Arity+k]])
                count = count + 1
            level_parents = ListOfNodes
        for c in level_parents:
            c.Childrens = [ FeatureList[LeafNo[0] + i] for i in range(Arity)]
            c.IsLeaf = True
            LeafNo[0] = LeafNo[0] + Arity 
        return node

def BinaryCreateEmptyTopGraph(Arity, Levels, Objective, FeatureList,Parent):
    if BinaryVerifyLevels(Levels,Arity) == False:
        sys.exit("Verify Levels of the Graph")
    else:
        node = Node(Arity,Objective,Parent)
        node.Childrens = [Node(Arity,Objective,node) for _ in range(Levels[1])]
        level_parents = node.Childrens
        for i in range(2,len(Levels)):
            ListOfNumbers = ListOfNumberGeneratorWithoutRepetition(Levels[i],int((Levels[i-1]*Arity)/Levels[i]))
            ListOfNodes = BinaryListOfNodesGenerator(Levels[i],Arity,Objective)
            count = 0
            for j in level_parents:
                for k in range(Arity):
                    ListOfNodes[ListOfNumbers[count*Arity+ k]].Parent.append(j)
                    j.Childrens.append(ListOfNodes[ListOfNumbers[count*Arity+k]])
                count = count + 1
            level_parents = ListOfNodes
        NbRepeat = int((Levels[len(Levels) -1] * Arity) / len(FeatureList))
        a = list(repeat(FeatureList,NbRepeat))
        listOfFeaturesRepeated = [j for i in a for j in i]
        count = 0
        for c in level_parents:
            c.IsLeaf = True 
            for b in range(Arity):
                c.Childrens.append(listOfFeaturesRepeated[count * Arity + b])
            count = count + 1
        return node

def CreateEmptyGraph(Arity, Levels, ObjectiveList, FeatureList, Parent,Target_sets_value):
    if VerifyLevels(Levels,Arity,ObjectiveList) == False:
        sys.exit("Verify Levels of the Graph")
    else:
        node = Node(len(ObjectiveList),[],Parent)
        node.Childrens = [Node(Arity,ObjectiveList[i],[node]) for i in range(Levels[1])]
        level_parents = node.Childrens
        for i in range(2,len(Levels)):
            ListOfNumbers = ListOfNumberGeneratorWithoutRepetition(Levels[i],int((Levels[i-1]*Arity)/Levels[i]))
            ListOfNodes = ListOfNodesGenerator(Levels[i],Arity,ObjectiveList,Target_sets_value)
            count = 0
            for j in level_parents:
                for k in range(Arity):
                    ListOfNodes[ListOfNumbers[count*Arity+ k]].Parent.append(j)
                    j.Childrens.append(ListOfNodes[ListOfNumbers[count*Arity+k]])
                count = count + 1
            level_parents = ListOfNodes
        for c in level_parents:
            c.Childrens = random.sample(FeatureList,Arity)
            c.IsLeaf = True
        return node
        
def CreateEmptyGraphAllFeature(Arity, Levels, ObjectiveList, FeatureList, Parent, LeafNo,Target_sets_value):
    if VerifyLevels(Levels,Arity,ObjectiveList) == False:
        sys.exit("Verify Levels of the Graph")
    else:
        node = Node(len(ObjectiveList),[],Parent)
        node.Childrens = [Node(Arity,ObjectiveList[i],[node]) for i in range(Levels[1])]
        level_parents = node.Childrens
        for i in range(2,len(Levels)):
            if i == 2:
                ListOfNumbers = ListOfNumberGeneratorWithoutRepetition(Levels[i],int((Levels[i-1]*Arity)/Levels[i]))
            else:
                ListOfNumbers = ListOfNumberGenerator(Levels[i],int((Levels[i-1]*Arity)/Levels[i]))
            ListOfNodes = ListOfNodesGenerator(Levels[i],Arity,ObjectiveList,Target_sets_value)
            count = 0
            for j in level_parents:
                for k in range(Arity):
                    ListOfNodes[ListOfNumbers[count*Arity+ k]].Parent.append(j)
                    j.Childrens.append(ListOfNodes[ListOfNumbers[count*Arity+k]])
                count = count + 1
            level_parents = ListOfNodes
        for c in level_parents:
            c.Childrens = [ FeatureList[LeafNo[0] + i] for i in range(Arity)]
            c.IsLeaf = True
            LeafNo[0] = LeafNo[0] + Arity 
        return node

def CreateEmptyTopGraph(Arity, Levels, ObjectiveList, FeatureList, Parent,Target_sets_value):
    if VerifyLevels(Levels,Arity,ObjectiveList) == False:
        sys.exit("Verify Levels of the Graph")
    else:
        node = Node(len(ObjectiveList),[],Parent)
        node.Childrens = [Node(Arity,ObjectiveList[i],[node]) for i in range(Levels[1])]
        level_parents = node.Childrens
        for i in range(2,len(Levels)):
            ListOfNumbers = ListOfNumberGeneratorWithoutRepetition(Levels[i],int((Levels[i-1]*Arity)/Levels[i]))
            ListOfNodes = ListOfNodesGenerator(Levels[i],Arity,ObjectiveList,Target_sets_value)
            count = 0
            for j in level_parents:
                for k in range(Arity):
                    ListOfNodes[ListOfNumbers[count*Arity+ k]].Parent.append(j)
                    j.Childrens.append(ListOfNodes[ListOfNumbers[count*Arity+k]])
                count = count + 1
            level_parents = ListOfNodes

        NbRepeat = int((Levels[len(Levels) -1] * Arity) / len(FeatureList))
        a = list(repeat(FeatureList,NbRepeat))
        listOfFeaturesRepeated = [j for i in a for j in i]
        shuffle(listOfFeaturesRepeated)
        count = 0
        for c in level_parents:
            c.IsLeaf = True 
            for b in range(Arity):
                c.Childrens.append(listOfFeaturesRepeated[count * Arity + b])
            count = count + 1
        return node

def BinaryCreateGreedyGraph(Arity, Levels, Objective, FeatureList,hd,Eps_top,Eps_bottom,neighborsList,choice,mask):
    head = BinaryCreateEmptyGraph(Arity, Levels, Objective, FeatureList,[])
    head.IsHead = True
    head.BinaryOptimizeTruthTableSubGraph(Levels,hd,Eps_top,Eps_bottom,neighborsList,choice,mask)
    return head

def BinaryCreateGreedyGraphAllFeatures(Arity, Levels, Objective, FeatureList,hd,Eps_top,Eps_bottom,neighborsList,choice,mask):
    head = BinaryCreateEmptyGraphAllFeature(Arity, Levels, Objective, FeatureList,[],[0])
    head.IsHead = True
    head.BinaryOptimizeTruthTableSubGraph(Levels,hd,Eps_top,Eps_bottom,neighborsList,choice,mask)
    return head

def BinaryCreateTopGreedyGraph(Arity, Levels, Objective, FeatureList,hd,Eps_top,Eps_bottom,neighborsList,choice,mask):
    head = BinaryCreateEmptyTopGraph(Arity, Levels, Objective, FeatureList,[])
    head.IsHead = True
    head.BinaryOptimizeTruthTableSubGraph(Levels,hd,Eps_top,Eps_bottom,neighborsList,choice,mask)
    return head

def CreateGreedyGraph(Arity, Levels, ObjectiveList, FeatureList,hd,Eps_top,Eps_bottom,neighborsList,Target_sets_value):
    head = CreateEmptyGraph(Arity, Levels, ObjectiveList, FeatureList,[],Target_sets_value)
    head.IsRoot = True
    for child in head.Childrens:
        child.IsHead = True
    head.OptimizeTruthTableSubGraph(Levels,hd,Eps_top,Eps_bottom,neighborsList)
    return head

def CreateGreedyGraphAllFeatures(Arity, Levels, ObjectiveList, FeatureList,hd,Eps_top,Eps_bottom,neighborsList,Target_sets_value):
    head = CreateEmptyGraphAllFeature(Arity, Levels, ObjectiveList, FeatureList,[],[0],Target_sets_value)
    head.IsRoot = True
    for child in head.Childrens:
        child.IsHead = True
    head.OptimizeTruthTableSubGraph(Levels,hd,Eps_top,Eps_bottom,neighborsList)
    return head

def CreateGreedyTopGraph(Arity, Levels, ObjectiveList, FeatureList,hd,Eps_top,Eps_bottom,neighborsList,Target_sets_value):
    head = CreateEmptyTopGraph(Arity, Levels, ObjectiveList, FeatureList,[],Target_sets_value)
    head.IsRoot = True
    for child in head.Childrens:
        child.IsHead = True
    head.OptimizeTruthTableSubGraph(Levels,hd,Eps_top,Eps_bottom,neighborsList)
    return head

class BinaryGraphClass:
    def __init__(self,Arity,Levels,FeatureList,Objective,x,neighborsList,choice,hd,Eps_top,Eps_bottom,mask):
        self.Arity = Arity
        self.Levels = Levels
        self.Mask = mask
        if x == 0:
            if len(FeatureList) == Arity * Levels[len(Levels)-1]:
                print("COMPLETE GRAPH")
                self.Head = BinaryCreateGreedyGraphAllFeatures(Arity, Levels, Objective, FeatureList,hd,Eps_top,Eps_bottom,neighborsList,choice,mask)
            else:
                self.Head = BinaryCreateGreedyGraph(Arity, Levels, Objective, FeatureList,hd,Eps_top,Eps_bottom,neighborsList,choice,mask)
        elif x == 1:
            self.Head = BinaryCreateGreedyGraph(Arity, Levels, Objective, FeatureList,hd,Eps_top,Eps_bottom,neighborsList,choice,mask)
        else:
            self.Head = BinaryCreateTopGreedyGraph(Arity, Levels, Objective, FeatureList,hd,Eps_top,Eps_bottom,neighborsList,choice,mask)

    def OptimizeLeafs(self,NbOpt,FeatureList,CorrelationMatrice,k,Levels,hd,Eps_top,Eps_bottom,neighborsList,choice,mask):
        for _ in range(NbOpt):
            self.OptimizeLeafWithCorrMatrice(FeatureList,CorrelationMatrice,k,Levels,hd,Eps_top,Eps_bottom,neighborsList,choice,mask)


    def OptimizeLeafWithCorrMatrice(self,FeatureList,CorrelationMatrice,k,Levels,hd,Eps_top,Eps_bottom,neighborsList,choice,mask):
        OldAccuracy = BinaryAccuracy(self.Head.Feature, self.Head.Objective)

        leaf = self.Head.RandomLeaf()
        FeatureNumber = random.choice(range(self.Head.Arity))
        OldFeature = leaf.Childrens[FeatureNumber]
        nbOldFeature = OldFeature.FeatureNb
        CandidateFeatures = [CorrelationMatrice[nbOldFeature][i] for i in range(k)]
        newFeatureNb = random.choice(range(k))
        leaf.Childrens[FeatureNumber] = CandidateFeatures[newFeatureNb][0]

        leaf.OptimizeTruthTableToTop(Levels,hd,Eps_top,Eps_bottom,neighborsList,choice,mask)
        NewAccuracy = BinaryAccuracy(self.Head.Feature, self.Head.Objective)
        if NewAccuracy[0] < OldAccuracy[0]:
            leaf.Childrens[FeatureNumber] = OldFeature
            leaf.OptimizeTruthTableToTop(Levels,hd,Eps_top,Eps_bottom,neighborsList,choice,mask)

    def OptimizeNode(self,hd,Eps_top,Eps_bottom,Arity,neighborsList,Prob_list,mask):
        level = numpy.random.choice(numpy.arange(1,len(self.Levels)+1),p = Prob_list)
        node = self.Head.RandomChild(level)
        #l = node.getLevel()
        if level == 1:
            list_d = ErrorCalculate(hd,Arity,Eps_top)
        elif level == len(self.Levels):
            list_d = ErrorCalculate(hd,Arity,Eps_bottom)
        else:
            l = len(self.Levels)
            Eps_interpolated = ((l - level) / ( l - 1)) * Eps_top + ((level - 1) / (l - 1)) * Eps_bottom
            list_d = ErrorCalculate(hd,Arity,Eps_interpolated)

        node.BinaryOptimizeFeatureFromTop(hd,list_d,neighborsList,mask)

    def OptimizeNodes(self, NbOpt,hd,Eps_top,Eps_bottom,Arity,neighborsList,Prob_list,mask):
        for _ in range(NbOpt):
            self.OptimizeNode(hd,Eps_top,Eps_bottom,Arity,neighborsList,Prob_list,mask)

    def Output(self):   # return the feature at the head.
        return self.Head.Feature

    def ObjOutput(self):
        return self.Head.Objective

    def getLeafNodes(self):
        children = self.Head.Childrens
        while children[0].IsLeaf == False:
            children = RetrieveNodeChildren(children)
        return children


class GraphClass:
    def __init__(self, Arity, Levels, FeatureList, ObjectiveList,x,hd,Eps_top,Eps_bottom,neighborsList,Target_sets_value):
        self.Arity = Arity
        self.Levels = Levels
        if x == 0:
            if len(FeatureList) == Arity * Levels[len(Levels)-1]:
                print("COMPLETE GRAPH")
                self.Head = CreateGreedyGraphAllFeatures(Arity, Levels, ObjectiveList, FeatureList,Levels,hd,Eps_top,Eps_bottom,neighborsList,Target_sets_value)
            else:
                self.Head = CreateGreedyGraph(Arity, Levels, ObjectiveList, FeatureList,hd,Eps_top,Eps_bottom,neighborsList,Target_sets_value)
        elif x == 1:
            self.Head = CreateGreedyGraph(Arity, Levels, ObjectiveList, FeatureList,hd,Eps_top,Eps_bottom,neighborsList,Target_sets_value)
        else:
            self.Head = CreateGreedyTopGraph(Arity, Levels, ObjectiveList, FeatureList,hd,Eps_top,Eps_bottom,neighborsList,Target_sets_value)

    def OptimizeNode(self,hd,Eps_top,Eps_bottom,Arity,neighborsList,c,Prob_list):
        level = numpy.random.choice(numpy.arange(2,len(self.Levels)+1),p = Prob_list)
        node = self.Head.RandomChild(level)

        if level == 2:
            list_d = ErrorCalculate(hd,Arity,Eps_top)
        elif level == len(self.Levels):
            list_d = ErrorCalculate(hd,Arity,Eps_bottom)
        else:
            l = len(self.Levels)
            Eps_interpolated = ((l - level) / ( l - 1)) * Eps_top + ((level - 1) / (l - 1)) * Eps_bottom
            list_d = ErrorCalculate(hd,Arity,Eps_interpolated)
        node.OptimizeFeatureFromTop(hd,list_d,neighborsList,c)

    def OptimizeNodes(self, NbOpt,hd,Eps_top,Eps_bottom,Arity,neighborsList,c,Prob_list):
        for _ in range(NbOpt):
            self.OptimizeNode(hd,Eps_top,Eps_bottom,Arity,neighborsList,c,Prob_list)
        
    def Output(self):   # return the feature at the head.
        self.Head.FeatureList = [i.Feature for i in self.Head.Childrens]
        return self.Head.FeatureList

    def getLeafNodes(self):
        children = self.Head.Childrens
        while children[0].IsLeaf == False:
            children = RetrieveNodeChildren(children)
        return children

    
def FOptimizeNodes(x):
    x[0].OptimizeNodes(x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8])
    return x[0]

def FOptimizeLeafs(x):
    x[0].OptimizeLeafs(x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9],x[10],x[11])
    return x[0]

def FGraphClass(x):
    return BinaryGraphClass(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9],x[10])
    

class ForestClass:
    #init: from a feature list build a Forest where the tree have been optimize
    #with the greedy algorithm
    def __init__(self,NbGraph, Arity, Levels, FeatureList, Objective,x,neighborsList,choice,hd,Eps_top,Eps_bottom,Prob_list,bagging):   
        self.Arity = Arity
        self.Levels = Levels
        l = FeatureList[0].NbTrain
        if bagging == 1:
            data = [ [Arity, Levels, FeatureList, Objective,x,neighborsList,choice,hd,Eps_top,Eps_bottom,random.randint(0,2 ** l - 1)] for _ in range(NbGraph)]
        else:
            data = [ [Arity, Levels, FeatureList, Objective,x,neighborsList,choice,hd,Eps_top,Eps_bottom,None] for _ in range(NbGraph)]
        self.Graphs = pool.map(FGraphClass,data)
        #self.Graphs = map(FGraphClass,data)   
        
    def MasksRetrieve(self):
        return [ i.Mask for i in self.Graphs] 

    def OptimizeLeafs(self,NbOpt,FeatureList,CorrelationMatrice,k,Levels,hd,Eps_top,Eps_bottom,neighborsList,choice,bagging):       # perform N leaf optimisation on every Graph
        if bagging == 1:
            data = [[x,NbOpt,FeatureList,CorrelationMatrice,k,Levels,hd,Eps_top,Eps_bottom,neighborsList,choice,x.Mask] for x in self.Graphs]
        else:
            data = [[x,NbOpt,FeatureList,CorrelationMatrice,k,Levels,hd,Eps_top,Eps_bottom,neighborsList,choice,None] for x in self.Graphs]
        self.Graphs = pool.map(FOptimizeLeafs,data)
        #self.Graphs = map(FOptimizeLeafs,data)
        
    def OptimizeNodes(self,NbOpt,hd,Eps_top,Eps_bottom,Arity,neighborsList,Prob_list,bagging):       # perform N node optimisation on every Graph
        if bagging == 1:
            data = [[x,NbOpt,hd,Eps_top,Eps_bottom,Arity,neighborsList,Prob_list,x.Mask] for x in self.Graphs]
        else:
            data = [[x,NbOpt,hd,Eps_top,Eps_bottom,Arity,neighborsList,Prob_list,None] for x in self.Graphs]
        self.Graphs = pool.map(FOptimizeNodes,data)
        #self.Graphs = map(FOptimizeNodes,data)
   
    def Output(self):               # return the list of features output (each tree)
        return [ x.Output() for x in self.Graphs]

    def ObjOutput(self):
        return [x.ObjOutput() for x in self.Graphs]

    def OutputHeads(self):
        return [x.Head for x in self.Graphs]

def bigGraphConstruct(listOfHeads,Graph,Arity):
    leafNodes = Graph.getLeafNodes()
    for c in listOfHeads:
        c.IsHead = False
    for i in leafNodes:
        i.IsLeaf = False
        for j in range(Arity):
            bool = False
            for k in range(len(listOfHeads)):
                if bool == False:
                    if i.Childrens[j] == listOfHeads[k].Feature:
                        bool = True
                        i.Childrens[j] = listOfHeads[k]
                        listOfHeads[k].Parent.append(i)
                else: 
                    break
    return Graph.Head

    
class multiForestClass:
    def __init__(self,NbGraph, Arity, Levels, FeatureList, ObjectiveList,x,neighborsList,choice,hd,Eps_top,Eps_bottom,Prob_list,bagging):
        self.Arity = Arity
        self.Levels = Levels
        l = FeatureList[0].NbTrain
        if bagging == 1:
            data = [ [Arity, Levels, FeatureList, ObjectiveList[i],x,neighborsList,choice,hd,Eps_top,Eps_bottom,random.randint(0,2 ** l - 1)] for i in range(len(ObjectiveList))]
        else:
            data = [ [Arity, Levels, FeatureList, ObjectiveList[i],x,neighborsList,choice,hd,Eps_top,Eps_bottom,None] for i in range(len(ObjectiveList))]
        self.Graphs = pool.map(FGraphClass,data)

    def MasksRetrieve(self):
        return [ i.Mask for i in self.Graphs]

    def OptimizeLeafs(self,NbOpt,FeatureList,CorrelationMatrice,k,Levels,hd,Eps_top,Eps_bottom,neighborsList,choice,bagging):       # perform N leaf optimisation on every Graph
        if bagging == 1:
            data = [[x,NbOpt,FeatureList,CorrelationMatrice,k,Levels,hd,Eps_top,Eps_bottom,neighborsList,choice,x.Mask] for x in self.Graphs]
        else:
            data = [[x,NbOpt,FeatureList,CorrelationMatrice,k,Levels,hd,Eps_top,Eps_bottom,neighborsList,choice,None] for x in self.Graphs]
        self.Graphs = pool.map(FOptimizeLeafs,data)
        #self.Graphs = map(FOptimizeLeafs,data)
        
    def OptimizeNodes(self,NbOpt,hd,Eps_top,Eps_bottom,Arity,neighborsList,Prob_list,bagging):       # perform N node optimisation on every Graph
        if bagging == 1:
            data = [[x,NbOpt,hd,Eps_top,Eps_bottom,Arity,neighborsList,Prob_list,x.Mask] for x in self.Graphs]
        else:
            data = [[x,NbOpt,hd,Eps_top,Eps_bottom,Arity,neighborsList,Prob_list,None] for x in self.Graphs]
        self.Graphs = pool.map(FOptimizeNodes,data)
        #self.Graphs = map(FOptimizeNodes,data)

    def Output(self):               # return the list of features output (each tree)
        return [ x.Output() for x in self.Graphs]

    def ObjOutput(self):
        return [x.ObjOutput() for x in self.Graphs]

    def OutputHeads(self):
        return [x.Head for x in self.Graphs]

def ObjectiveListConstruct(ObjectiveList,comb):
    ObjectiveListAllCombinations = []
    a = [0,1,2,3,4,5,6,7,8,9]
    b = list(combinations(a,comb))
    for i in range(len(b)):
        lst = []
        for j in range(len(b[0])):
            lst.append(ObjectiveList[b[i][j]])
        ObjectiveListAllCombinations.append(lst)
    return ObjectiveListAllCombinations

def OrBetweenObjectives(ObjectiveList):
    newObjective = ObjectiveClass([0],[0],11)
    objTrain = [i.TrainOne for i in ObjectiveList]
    trainValue = functools.reduce(operator.ior,objTrain)
    negobjTrain = [i.TrainZero for i in ObjectiveList]
    negTrainValue = functools.reduce(operator.iand,negobjTrain)
    objTest = [i.TestOne for i in ObjectiveList]
    testValue = functools.reduce(operator.ior,objTest)
    negobjtest = [i.TestZero for i in ObjectiveList]
    negTestValue = functools.reduce(operator.iand,negobjtest)
    newObjective.TrainOne = trainValue
    newObjective.TrainZero = negTrainValue
    newObjective.TestOne = testValue
    newObjective.TestZero = negTestValue
    newObjective.NbTrain = ObjectiveList[0].NbTrain
    newObjective.NbTest = ObjectiveList[0].NbTest
    return newObjective

def OrObjectives(ObjectiveList):
    new_ObjectiveList = []
    for i in ObjectiveList:
        new_ObjectiveList.append(OrBetweenObjectives(i))
    return new_ObjectiveList

class AllMultiGraphClass:
    def __init__(self,listOfHeads,Graph,Arity,NbLevels):
        self.Arity = Arity
        self.NbLevels = NbLevels
        self.Head = bigGraphConstruct(listOfHeads,Graph,Arity)

    def OptimizeNode(self,hd,Eps_top,Eps_bottom,Arity,neighborsList,c,Prob_list):
        level = numpy.random.choice(numpy.arange(2,self.NbLevels+1),p = Prob_list) #arange(1, as binaryGraph levels begins by 1 for multiclass it is 2
        node = self.Head.RandomChild(level)
        if level == 1:
            list_d = ErrorCalculate(hd,Arity,Eps_top)
        elif level == self.NbLevels:
            list_d = ErrorCalculate(hd,Arity,Eps_bottom)
        else:
            l = self.NbLevels
            Eps_interpolated = ((l - level) / ( l - 1)) * Eps_top + ((level - 1) / (l - 1)) * Eps_bottom
            list_d = ErrorCalculate(hd,Arity,Eps_interpolated)
        node.OptimizeFeatureFromTop(hd,list_d,neighborsList,c)

    def OptimizeNodes(self,NbOpt,hd,Eps_top,Eps_bottom,Arity,neighborsList,c,Prob_list):
        for _ in range(NbOpt):
            self.OptimizeNode(hd,Eps_top,Eps_bottom,Arity,neighborsList,c,Prob_list)

    def Output(self):   # return the feature at the head.
        self.Head.FeatureList = [i.Feature for i in self.Head.Childrens]
        return self.Head.FeatureList


class AllGraphClass:
    def __init__(self,listOfHeads,Graph,Arity,NbLevels):
        self.Arity = Arity
        self.NbLevels = NbLevels
        self.Head = bigGraphConstruct(listOfHeads,Graph,Arity)

    def OptimizeNode(self,hd,Eps_top,Eps_bottom,Arity,neighborsList,Prob_list):
        level = numpy.random.choice(numpy.arange(1,self.NbLevels+1),p = Prob_list) #arange(1, as binaryGraph levels begins by 1 for multiclass it is 2
        node = self.Head.RandomChild(level)
        if level == 1:
            list_d = ErrorCalculate(hd,Arity,Eps_top)
        elif level == self.NbLevels:
            list_d = ErrorCalculate(hd,Arity,Eps_bottom)
        else:
            l = self.NbLevels
            Eps_interpolated = ((l - level) / ( l - 1)) * Eps_top + ((level - 1) / (l - 1)) * Eps_bottom
            list_d = ErrorCalculate(hd,Arity,Eps_interpolated)
        node.BinaryOptimizeFeatureFromTop(hd,list_d,neighborsList,None)

    def OptimizeNodes(self,NbOpt,hd,Eps_top,Eps_bottom,Arity,neighborsList,Prob_list):
        for _ in range(NbOpt):
            self.OptimizeNode(hd,Eps_top,Eps_bottom,Arity,neighborsList,Prob_list)

    def Output(self):
        return self.Head.Feature


def EliminateLowEntropyFeature(FeatureList, Threshold):
    # remove feature with entropy below the threshold
    FeaturesListCopy = copy.copy(FeatureList)
    for i in range(len(FeatureList)):
        nbOnesFract = popcount(FeatureList[i].TrainValue)/FeatureList[0].NbTrain
        if H(nbOnesFract) <= Threshold:
            FeaturesListCopy.remove(FeatureList[i])
    return FeaturesListCopy

def EliminateRedundantFeature(FeatureList, Threshold): 
    # remove feature for which there is another feature within the threshold
    # with higher entropy
    return FeatureList

def CleanFeatureList(FeatureList, MinEntropy, MinSimilarity):
    return EliminateRedundantFeature(EliminateLowEntropyFeature(FeatureList, MinEntropy), MinSimilarity)

def Compress(FeatureList):
    FeatureListOutput = FeatureList
    #calculate Feature
    return FeatureList2

def MajorityFeature(ForestFeatureList):
    # translate features into bit list
    # for every bit compute the majority
    # convert into feature again
    return Feature


####################################################
# Test functions
def Extract(DataSet):
    DataT = [ IntToBitList(x.TrainValue,x.NbTrain) for x in DataSet.Features]
    DataTrain = Transpose(DataT)
    TargetTrain = IntToBitList(DataSet.Objective.TrainOne, DataSet.Objective.NbTrain)
    for n in range(3):
        print("Target = ",TargetTrain[n])
        AfficheTextImage(DataTrain[n])
    DataT = [ IntToBitList(x.TestValue,x.NbTest) for x in DataSet.Features]
    DataTest = Transpose(DataT)
    
    print("=========================================")
    TargetTest = IntToBitList(DataSet.Objective.TestOne, DataSet.Objective.NbTest)
    for n in range(3):
        print("Target = ",TargetTest[n])
        AfficheTextImage(DataTest[n])
##############################################################################
##############################################################################
##############################################################################
# Clasification algorithm
def Maj3(x,y,z):
    out = copy.copy(x)
    

    out.TrainValue = x.TrainValue & y.TrainValue | x.TrainValue & z.TrainValue | y.TrainValue & z.TrainValue
    out.NegTrainValue = out.TrainValue ^  x.TrainValue^x.NegTrainValue

    out.TestValue = x.TestValue & y.TestValue | x.TestValue & z.TestValue | y.TestValue & z.TestValue
    out.NegTestValue = out.TestValue ^  x.TestValue ^ x.NegTestValue

    return out

def MajP3(l):
    le = len(l)
    le3=int(le/3)
    if le == 3:
        return Maj3(l[0],l[1],l[2])
    else:
        return Maj3( MajP3(l[0:le3]), MajP3(l[le3:2 * le3]), MajP3(l[2 * le3:]) )

def forFinalEvaluation(FeatureList,ObjectiveList):
    x = Transpose([i.FeatureToListForTrain() for i in FeatureList])
    list1 = [BitListToInt(i) for i in x]
    y = Transpose([i.FeatureToListForTest() for i in FeatureList])
    list2 = [BitListToInt(i) for i in y]
    z = Transpose([i.ObjectiveToListTrain() for i in ObjectiveList])
    list3 = [BitListToInt(i) for i in z]
    w = Transpose([i.ObjectiveToListTest() for i in ObjectiveList])
    list4 = [BitListToInt(i) for i in w]
    return list1,list2,list3,list4

def createVectorOf_0_1(NbExples,set_card):
    nb_zero = NbExples - set_card
    vect = [0 for _ in range(nb_zero)]
    ones = [1 for _ in range(set_card)]
    vect.extend(ones)
    random.shuffle(vect)
    vector = BitListToInt(vect)
    return vector

def BaggingObjective(Objective,Train_set_card,Test_set_card):
    NewTargetTrainValue = createVectorOf_0_1(Objective.NbTrain,Train_set_card) & Objective.TrainOne
    NewTargetNegTrainValue = NegInt(NewTargetTrainValue,Objective.NbTrain)
    NewTargetTestValue = createVectorOf_0_1(Objective.NbTest,Test_set_card) & Objective.TestOne
    NewTargetNegTestValue = NegInt(NewTargetTestValue,Objective.NbTest)
    NewTarget = ObjectiveClass([0],[0],Objective.NbObjective)
    NewTarget.NbTrain = Objective.NbTrain
    NewTarget.NbTest = Objective.NbTest
    NewTarget.TrainOne = NewTargetTrainValue
    NewTarget.TrainZero = NewTargetNegTrainValue
    NewTarget.TestOne = NewTargetTestValue
    NewTarget.TestZero = NewTargetNegTestValue
    return NewTarget


##############################################################################

def GraphAlgorithm(Arity, Levels,FeatureList,ObjectiveList,NbLeafOpt,LeafOptStp,NbNodeOpt,NodeOptStp,k,c0, c1, c2,hd,Eps_top,Eps_bottom,c,Levels_Prob,Target_sets_value):

    neighborsList = calculateNeighbors(Arity)

    tot = sum(Levels_Prob)
    PList = [i/tot for i in Levels_Prob]
    Prob_list = numpy.asarray(PList)

    Graph = GraphClass(Arity,Levels,FeatureList,ObjectiveList, 0, c0, c1, c2, neighborsList,Target_sets_value)
    print("Greedy")
    headFeaturesTrain,headFeaturesTest,TargetsTrain,TargetsTest = forFinalEvaluation(Graph.Output(),ObjectiveList) 
    PrintAccuracy(headFeaturesTrain,headFeaturesTest,TargetsTrain,TargetsTest)


    for i in range(NbNodeOpt):
        Graph.OptimizeNodes(NodeOptStp,hd,Eps_top,Eps_bottom,Arity,neighborsList,c,Prob_list)
        print("Nodes ",i + 1)
        headFeaturesTrain1,headFeaturesTest1,TargetsTrain1,TargetsTest1 = forFinalEvaluation(Graph.Output(),ObjectiveList) 
        PrintAccuracy(headFeaturesTrain1,headFeaturesTest1,TargetsTrain1,TargetsTest1)

def ParamPrint(Arity,TopLevels,Levels,NbG_all_all,NbG_one_all, FeatureList,ObjectiveList,NbLeafOpt,LeafOptStp,NbNodeOpt_Step_1,NodeOptStp_Step_1,NbNodeOpt_Step_2,NodeOptStp_Step_2,NbNodeOpt_Step_3,NodeOptStp_Step_3,k,hd,c,Eps_top_1,Eps_bottom_1,Eps_top_2,Eps_bottom_2,Eps_top_3,Eps_bottom_3,BottomLevels_Prob,TopLevels_Prob,AllLevels_Prob,choice,Target_sets_value):
    print("Arity = ", Arity)
    print("TopLevels = ",TopLevels)
    print("Levels = ", Levels)
    print("Nb_One_All = ",NbG_one_all)
    print("Nb_All_All = ",NbG_all_all)
    print("NbLeafOpt = ",NbLeafOpt," * ",LeafOptStp)
    print("NbNodeOpt_1 = ",NbNodeOpt_Step_1," * ",NodeOptStp_Step_1)
    print("NbNodeOpt_2 = ",NbNodeOpt_Step_2," * ",NodeOptStp_Step_2)
    print("NbNodeOpt_3 = ",NbNodeOpt_Step_3," * ",NodeOptStp_Step_3)
    print("NbNeighbors = ", k)
    print("Hamming distance = ",hd)
    print("Eps_top_1 = ",Eps_top_1,"Eps_bottom_1 = ",Eps_bottom_1)
    print("Eps_top_2 = ",Eps_top_2,"Eps_bottom_2 = ",Eps_bottom_2)
    print("Eps_top_3 = ",Eps_top_3,"Eps_bottom_3 = ",Eps_bottom_3)
    print("SubGraphs_Levels_Prob = ",BottomLevels_Prob)
    print("TopGraph_Levels_Prob  = ",TopLevels_Prob)
    print("BigGraph_Levels_Prob = ",AllLevels_Prob)
    print("Info_Or_Prob = ",choice)
    print("Target_Sets_Or_Value = ",Target_sets_value,"\r\n")

def BinaryParamPrint(Arity,TopLevels,Levels,NbGraph,NbLeafOpt,LeafOptStp,NbNodeOpt_Step_1,NodeOptStp_Step_1,NbNodeOpt_Step_2,NodeOptStp_Step_2,NbNodeOpt_Step_3,NodeOptStp_Step_3,k,hd,Eps_top_1,Eps_bottom_1,Eps_top_2,Eps_bottom_2,Eps_top_3,Eps_bottom_3,BottomLevels_Prob,TopLevels_Prob,AllLevels_Prob,choice):
    print("Arity = ", Arity)
    print("TopLevels = ",TopLevels)
    print("Levels = ", Levels)
    print("NbGraph = ",NbGraph)
    print("NbLeafOpt = ",NbLeafOpt," * ",LeafOptStp)
    print("NbNodeOpt_1 = ",NbNodeOpt_Step_1," * ",NodeOptStp_Step_1)
    print("NbNodeOpt_2 = ",NbNodeOpt_Step_2," * ",NodeOptStp_Step_2)
    print("NbNodeOpt_3 = ",NbNodeOpt_Step_3," * ",NodeOptStp_Step_3)
    print("NbNeighbors = ", k)
    print("Hamming distance = ",hd)
    print("Eps_top_1 = ",Eps_top_1,"Eps_bottom_1 = ",Eps_bottom_1)
    print("Eps_top_2 = ",Eps_top_2,"Eps_bottom_2 = ",Eps_bottom_2)
    print("Eps_top_3 = ",Eps_top_3,"Eps_bottom_3 = ",Eps_bottom_3)
    print("SubGraphs_Levels_Prob = ",BottomLevels_Prob)
    print("TopGraph_Levels_Prob  = ",TopLevels_Prob)
    print("BigGraph_Levels_Prob = ",AllLevels_Prob)
    print("Info_Or_Prob = ",choice,"\r\n")


def BinaryGraph(Arity,TopLevels,Levels,NbGraph,FeatureList,Objective,NbLeafOpt,LeafOptStp,NbNodeOpt_Step_1,NodeOptStp_Step_1,NbNodeOpt_Step_2,NodeOptStp_Step_2,NbNodeOpt_Step_3,NodeOptStp_Step_3,k,CorrelationMatrice,hd,Eps_top_1,Eps_bottom_1,Eps_top_2,Eps_bottom_2,Eps_top_3,Eps_bottom_3,BottomLevels_Prob,TopLevels_Prob,AllLevels_Prob,choice,bagging = 1):

    print("\r\n")
    print(datetime.now().strftime("%Y-%m-%d %H:%M"),"\r\n")
    BinaryParamPrint(Arity,TopLevels,Levels,NbGraph,NbLeafOpt,LeafOptStp,NbNodeOpt_Step_1,NodeOptStp_Step_1,NbNodeOpt_Step_2,NodeOptStp_Step_2,NbNodeOpt_Step_3,NodeOptStp_Step_3,k,hd,Eps_top_1,Eps_bottom_1,Eps_top_2,Eps_bottom_2,Eps_top_3,Eps_bottom_3,BottomLevels_Prob,TopLevels_Prob,AllLevels_Prob,choice)

    neighborsList = calculateNeighbors(Arity)

    def Prob(Prob_l):
        tot = sum(Prob_l)
        PList = [i/tot for i in Prob_l]
        list_prob = numpy.asarray(PList)
        return list_prob

    def statics(NbGr,FeatureOutput,ObjectiveOutput):
                Accuracy_list = []
                for j in range(NbGr):
                    acc = BinaryAccuracy(FeatureOutput[j],ObjectiveOutput[j])
                    Accuracy_list.append(acc)
                print("max = ",max(Accuracy_list))
                print("min = ",min(Accuracy_list))
                t = Transpose(Accuracy_list)
                print("mean = ", [round(sum(t[0]) / float(len(t[0])),4),round(sum(t[1]) / float(len(t[1])),4)])

    def BaggingStatics(NbGr,FeatureOutput,ObjectiveOutput,MaskList):
        Accuracy_list = []
        for j in range(NbGr):
            acc = BaggingBinaryAccuracy(MaskList[j],FeatureOutput[j],ObjectiveOutput[j])
            Accuracy_list.append(acc)
        print("max = ",max(Accuracy_list))
        print("min = ",min(Accuracy_list))
        t = Transpose(Accuracy_list)
        print("mean = ", [round(sum(t[0]) / float(len(t[0])),4),round(sum(t[1]) / float(len(t[1])),4)])

    
    Prob_list = Prob(BottomLevels_Prob)
    TopProb_list = Prob(TopLevels_Prob)
    AllProb_list = Prob(AllLevels_Prob)

    if len(TopLevels) == 0:
        BGraph = BinaryGraphClass(Arity,Levels,FeatureList,Objective,0,neighborsList,choice,hd,Eps_top_1,Eps_bottom_1)
        print("Greedy")
        BinaryPrintAccuracy(BGraph.Output(),Objective)

        for i in range(NbLeafOpt):
            BGraph.OptimizeLeafs(LeafOptStp,FeatureList,CorrelationMatrice,k,Levels,hd,Eps_top_1,Eps_bottom_1,neighborsList,choice)
            print("Leafs ",i + 1)
            BinaryPrintAccuracy(BGraph.Output(),Objective)

        for i in range(NbNodeOpt_Step_1):
            BGraph.OptimizeNodes(NodeOptStp_Step_1,hd,Eps_top_1,Eps_bottom_1,Arity,neighborsList,Prob_list)
            print("Nodes ",i + 1)
            BinaryPrintAccuracy(BGraph.Output(),Objective)
    else:
        if bagging == 1:
            Forest = ForestClass(NbGraph,Arity,Levels,FeatureList,Objective,1,neighborsList,choice,hd,Eps_top_1,Eps_bottom_1,Prob_list,1)
            BaggingStatics(NbGraph,Forest.Output(),Forest.ObjOutput(),Forest.MasksRetrieve())

            for i in range(NbLeafOpt):
                Forest.OptimizeLeafs(LeafOptStp,FeatureList,CorrelationMatrice,k,Levels,hd,Eps_top_1,Eps_bottom_1,neighborsList,choice,1)
                print("Leafs ",i + 1)
                BaggingStatics(NbGraph,Forest.Output(),Forest.ObjOutput(),Forest.MasksRetrieve())
                print("-----------------")

            for i in range(NbNodeOpt_Step_1):
                Forest.OptimizeNodes(NodeOptStp_Step_1,hd,Eps_top_1,Eps_bottom_1,Arity,neighborsList,Prob_list,1)
                print("Nodes ",i + 1)
                BaggingStatics(NbGraph,Forest.Output(),Forest.ObjOutput(),Forest.MasksRetrieve())
                print("-----------------")
        else:
            Forest = ForestClass(NbGraph,Arity,Levels,FeatureList,Objective,1,neighborsList,choice,hd,Eps_top_1,Eps_bottom_1,Prob_list,0)
            print("Greedy ")
            statics(NbGraph,Forest.Output(),Forest.ObjOutput())

            for i in range(NbLeafOpt):
                Forest.OptimizeLeafs(LeafOptStp,FeatureList,CorrelationMatrice,k,Levels,hd,Eps_top_1,Eps_bottom_1,neighborsList,choice,0)
                print("Leafs ",i + 1)
                statics(NbGraph,Forest.Output(),Forest.ObjOutput())
                print("-----------------")

            for i in range(NbNodeOpt_Step_1):
                Forest.OptimizeNodes(NodeOptStp_Step_1,hd,Eps_top_1,Eps_bottom_1,Arity,neighborsList,Prob_list,0)
                print("Nodes ",i + 1)
                statics(NbGraph,Forest.Output(),Forest.ObjOutput())
                print("-----------------")

        Output = Forest.Output()
        BinaryFinalGraph = BinaryGraphClass(Arity,TopLevels,Output,Objective,2,neighborsList,choice,hd,Eps_top_2,Eps_bottom_2,None)
        BinaryPrintAccuracy(BinaryFinalGraph.Output(),Objective)
        for i in range(NbNodeOpt_Step_2):
            BinaryFinalGraph.OptimizeNodes(NodeOptStp_Step_2,hd,Eps_top_2,Eps_bottom_2,Arity,neighborsList,TopProb_list,None)
            print("Nodes ",i + 1)
            BinaryPrintAccuracy(BinaryFinalGraph.Output(),Objective)

        NbLevels = len(TopLevels) + len(Levels)
        headNodes = Forest.OutputHeads()
        BigGraph = AllGraphClass(headNodes,BinaryFinalGraph,Arity,NbLevels)
        BinaryPrintAccuracy(BigGraph.Output(),Objective)
        for i in range(NbNodeOpt_Step_3):
            BigGraph.OptimizeNodes(NodeOptStp_Step_3,hd,Eps_top_3,Eps_bottom_3,Arity,neighborsList,AllProb_list)
            print("Nodes ",i + 1)
            BinaryPrintAccuracy(BigGraph.Output(),Objective)

        print("\r\n")
        print(datetime.now().strftime("%Y-%m-%d %H:%M"),"\r\n")

def MultiClassGraph(Arity,TopLevels,Levels,NbG_all_all,NbG_one_all, FeatureList,ObjectiveList,NbLeafOpt,LeafOptStp,NbNodeOpt_Step_1,NodeOptStp_Step_1,NbNodeOpt_Step_2,NodeOptStp_Step_2,NbNodeOpt_Step_3,NodeOptStp_Step_3,k,CorrelationMatrice,hd,c,Eps_top_1,Eps_bottom_1,Eps_top_2,Eps_bottom_2,Eps_top_3,Eps_bottom_3,BottomLevels_Prob,TopLevels_Prob,AllLevels_Prob,choice,Target_sets_value,bagging):
    
    print("\r\n")
    print(datetime.now().strftime("%Y-%m-%d %H:%M"),"\r\n")
    ParamPrint(Arity,TopLevels,Levels,NbG_all_all,NbG_one_all, FeatureList,ObjectiveList,NbLeafOpt,LeafOptStp,NbNodeOpt_Step_1,NodeOptStp_Step_1,NbNodeOpt_Step_2,NodeOptStp_Step_2,NbNodeOpt_Step_3,NodeOptStp_Step_3,k,hd,c,Eps_top_1,Eps_bottom_1,Eps_top_2,Eps_bottom_2,Eps_top_3,Eps_bottom_3,BottomLevels_Prob,TopLevels_Prob,AllLevels_Prob,choice,Target_sets_value)

    def Prob(Prob_l):
            tot = sum(Prob_l)
            PList = [i/tot for i in Prob_l]
            list_prob = numpy.asarray(PList)
            return list_prob

    def statics(start,stop,FeatureOutput,ObjectiveOutput,masksList):
                Accuracy_list_1 = []
                Accuracy_list_0 = []
                for j in range(start,stop):
                    if masksList == None:
                        acc = BinaryAccuracy1(FeatureOutput[j],ObjectiveOutput[j])
                    else:
                        acc = BaggingBinaryAccuracy1(FeatureOutput[j],ObjectiveOutput[j],masksList[j])
                    Accuracy_list_1.append(acc[0])
                    Accuracy_list_0.append(acc[1])
                print("max_1 = ",max(Accuracy_list_1))
                print("max_0 = ",max(Accuracy_list_0))
                print("min_1 = ",min(Accuracy_list_1))
                print("min_0 = ",min(Accuracy_list_0))
                t_1 = Transpose(Accuracy_list_1)
                print("mean_1 = ", [round(sum(t_1[0]) / float(len(t_1[0])),4),round(sum(t_1[1]) / float(len(t_1[1])),4)])
                t_0 = Transpose(Accuracy_list_0)
                print("mean_0 = ", [round(sum(t_0[0]) / float(len(t_0[0])),4),round(sum(t_0[1]) / float(len(t_0[1])),4)])

    def Construct_One_All(ListOfObjectives,NbG):
        Objectives_one_all = ObjectiveListConstruct(ListOfObjectives,9)
        if NbG > 10:
            ObjGraph_one_all = random.sample(Objectives_one_all,10)
            NbG_one_all_rest = NbG - 10
            rest = [random.choice(ObjGraph_one_all) for _ in range(NbG_one_all_rest)]
            ObjGraph_one_all.extend(rest)
        else:
            ObjGraph_one_all = random.sample(Objectives_one_all,NbG)
        ObjectivesOneAll = OrObjectives(ObjGraph_one_all)
        return ObjectivesOneAll

    def Construct_All_All(ListOfObjectives,NbG):
        Objectives_all_all = ObjectiveListConstruct(ListOfObjectives,5)
        ObjGraph_all_all = random.sample(Objectives_all_all,NbG)
        Obj_all_all = OrObjectives(ObjGraph_all_all)
        return Obj_all_all

    NbG = NbG_all_all + NbG_one_all
    TopGraphLeavesNb = TopLevels[len(TopLevels) - 1]
    if (TopGraphLeavesNb * Arity) % NbG != 0:
        sys.exit("Verify Number of Subgraphs") 
    else:
        neighborsList = calculateNeighbors(Arity)
        if NbG_all_all == 0:
            Objectives = Construct_One_All(ObjectiveList,NbG_one_all)
        elif NbG_one_all == 0:
            Objectives = Construct_All_All(ObjectiveList,NbG_all_all)
        else:
            Objectives = Construct_One_All(ObjectiveList,NbG_one_all)
            All_All = Construct_All_All(ObjectiveList,NbG_all_all)
            Objectives.extend(All_All)

        Prob_list = Prob(BottomLevels_Prob)
        TopProb_list = Prob(TopLevels_Prob)
        AllProb_list = Prob(AllLevels_Prob)
    
        if len(TopLevels) == 0:
            Graph = GraphClass(Arity,Levels,FeatureList,ObjectiveList, 0, hd,Eps_top_1,Eps_bottom_2,neighborsList,Target_sets_value)
            print("Greedy")
            headFeaturesTrain,headFeaturesTest,TargetsTrain,TargetsTest = forFinalEvaluation(Graph.Output(),ObjectiveList) 
            PrintAccuracy(headFeaturesTrain,headFeaturesTest,TargetsTrain,TargetsTest)


            for i in range(NbNodeOpt_Step_1):
                Graph.OptimizeNodes(NodeOptStp_Step_1,hd,Eps_top_1,Eps_bottom_1,Arity,neighborsList,c,Prob_list)
                print("Nodes ",i + 1)
                headFeaturesTrain1,headFeaturesTest1,TargetsTrain1,TargetsTest1 = forFinalEvaluation(Graph.Output(),ObjectiveList) 
                PrintAccuracy(headFeaturesTrain1,headFeaturesTest1,TargetsTrain1,TargetsTest1)
        else:
            if bagging == 1:
                Forest = multiForestClass(NbG,Arity,Levels,FeatureList,Objectives,1,neighborsList,choice,hd,Eps_top_1,Eps_bottom_1,Prob_list,1)
                if NbG_all_all == 0:
                    print("One_All")
                    statics(0,NbG_one_all,Forest.Output(),Forest.ObjOutput(),Forest.MasksRetrieve())
                    print("-----------------")
                elif NbG_one_all == 0:
                    print("All_All")
                    statics(NbG_one_all,NbG,Forest.Output(),Forest.ObjOutput(),Forest.MasksRetrieve())
                    print("-----------------")
                else:
                    print("One_All")
                    statics(0,NbG_one_all,Forest.Output(),Forest.ObjOutput(),Forest.MasksRetrieve())
                    print("All_All")
                    statics(NbG_one_all,NbG,Forest.Output(),Forest.ObjOutput(),Forest.MasksRetrieve())
                    print("-----------------")

                for i in range(NbLeafOpt):
                    Forest.OptimizeLeafs(LeafOptStp,FeatureList,CorrelationMatrice,k,Levels,hd,Eps_top_1,Eps_bottom_1,neighborsList,choice,1)
                    print("Leafs",i + 1)
                    if NbG_all_all == 0:
                        print("One_All")
                        statics(0,NbG_one_all,Forest.Output(),Forest.ObjOutput(),Forest.MasksRetrieve())
                        print("-----------------")
                    elif NbG_one_all == 0:
                        print("All_All")
                        statics(NbG_one_all,NbG,Forest.Output(),Forest.ObjOutput(),Forest.MasksRetrieve())
                        print("-----------------")
                    else:
                        print("One_All")
                        statics(0,NbG_one_all,Forest.Output(),Forest.ObjOutput(),Forest.MasksRetrieve())
                        print("All_All")
                        statics(NbG_one_all,NbG,Forest.Output(),Forest.ObjOutput(),Forest.MasksRetrieve())
                        print("-----------------")

                for i in range(NbNodeOpt_Step_1):
                    Forest.OptimizeNodes(NodeOptStp_Step_1,hd,Eps_top_1,Eps_bottom_1,Arity,neighborsList,Prob_list,1)
                    print("Nodes",i + 1)
                    if NbG_all_all == 0:
                        print("One_All")
                        statics(0,NbG_one_all,Forest.Output(),Forest.ObjOutput(),Forest.MasksRetrieve())
                        print("-----------------")
                    elif NbG_one_all == 0:
                        print("All_All")
                        statics(NbG_one_all,NbG,Forest.Output(),Forest.ObjOutput(),Forest.MasksRetrieve())
                        print("-----------------")
                    else:
                        print("One_All")
                        statics(0,NbG_one_all,Forest.Output(),Forest.ObjOutput(),Forest.MasksRetrieve())
                        print("All_All")
                        statics(NbG_one_all,NbG,Forest.Output(),Forest.ObjOutput(),Forest.MasksRetrieve())
                        print("-----------------")
            else:
                Forest = multiForestClass(NbG,Arity,Levels,FeatureList,Objectives,1,neighborsList,choice,hd,Eps_top_1,Eps_bottom_1,Prob_list,0)
                if NbG_all_all == 0:
                    print("One_All")
                    statics(0,NbG_one_all,Forest.Output(),Forest.ObjOutput(),None)
                    print("-----------------")
                elif NbG_one_all == 0:
                    print("All_All")
                    statics(NbG_one_all,NbG,Forest.Output(),Forest.ObjOutput(),None)
                    print("-----------------")
                else:
                    print("One_All")
                    statics(0,NbG_one_all,Forest.Output(),Forest.ObjOutput(),None)
                    print("All_All")
                    statics(NbG_one_all,NbG,Forest.Output(),Forest.ObjOutput(),None)
                    print("-----------------")

            
                for i in range(NbLeafOpt):
                    Forest.OptimizeLeafs(LeafOptStp,FeatureList,CorrelationMatrice,k,Levels,hd,Eps_top_1,Eps_bottom_1,neighborsList,choice,0)
                    print("Leafs",i + 1)
                    if NbG_all_all == 0:
                        print("One_All")
                        statics(0,NbG_one_all,Forest.Output(),Forest.ObjOutput(),None)
                        print("-----------------")
                    elif NbG_one_all == 0:
                        print("All_All")
                        statics(NbG_one_all,NbG,Forest.Output(),Forest.ObjOutput(),None)
                        print("-----------------")
                    else:
                        print("One_All")
                        statics(0,NbG_one_all,Forest.Output(),Forest.ObjOutput(),None)
                        print("All_All")
                        statics(NbG_one_all,NbG,Forest.Output(),Forest.ObjOutput(),None)
                        print("-----------------")

                for i in range(NbNodeOpt_Step_1):
                    Forest.OptimizeNodes(NodeOptStp_Step_1,hd,Eps_top_1,Eps_bottom_1,Arity,neighborsList,Prob_list,0)
                    print("Nodes",i + 1)
                    if NbG_all_all == 0:
                        print("One_All")
                        statics(0,NbG_one_all,Forest.Output(),Forest.ObjOutput(),None)
                        print("-----------------")
                    elif NbG_one_all == 0:
                        print("All_All")
                        statics(NbG_one_all,NbG,Forest.Output(),Forest.ObjOutput(),None)
                        print("-----------------")
                    else:
                        print("One_All")
                        statics(0,NbG_one_all,Forest.Output(),Forest.ObjOutput(),None)
                        print("All_All")
                        statics(NbG_one_all,NbG,Forest.Output(),Forest.ObjOutput(),None)
                        print("-----------------")

            OutputFeatures = Forest.Output()
            FinalMultiClassGraph = GraphClass(Arity,TopLevels,OutputFeatures,ObjectiveList,2,hd,Eps_top_2,Eps_bottom_2,neighborsList,Target_sets_value)
            #sys.stdout = open('exp_arity_4', 'w')
            headFeaturesTrain1,headFeaturesTest1,TargetsTrain1,TargetsTest1 = forFinalEvaluation(FinalMultiClassGraph.Output(),ObjectiveList) 
            PrintAccuracy(headFeaturesTrain1,headFeaturesTest1,TargetsTrain1,TargetsTest1)
            print("------------------------------------------------")
            for i in range(NbNodeOpt_Step_2):
                FinalMultiClassGraph.OptimizeNodes(NodeOptStp_Step_2,hd,Eps_top_2,Eps_bottom_2,Arity,neighborsList,c,TopProb_list)
                print("Nodes",i + 1)
                headFeaturesTrain1,headFeaturesTest1,TargetsTrain1,TargetsTest1 = forFinalEvaluation(FinalMultiClassGraph.Output(),ObjectiveList) 
                PrintAccuracy(headFeaturesTrain1,headFeaturesTest1,TargetsTrain1,TargetsTest1)

            nbLevel = len(TopLevels) + len(Levels)
            heads = Forest.OutputHeads()
            BigGraph = AllMultiGraphClass(heads,FinalMultiClassGraph,Arity,nbLevel)
            headFeaturesTrain1,headFeaturesTest1,TargetsTrain1,TargetsTest1 = forFinalEvaluation(BigGraph.Output(),ObjectiveList) 
            PrintAccuracy(headFeaturesTrain1,headFeaturesTest1,TargetsTrain1,TargetsTest1)

            for i in range(NbNodeOpt_Step_3):
                BigGraph.OptimizeNodes(NodeOptStp_Step_3,hd,Eps_top_3,Eps_bottom_3,Arity,neighborsList,c,AllProb_list)
                print("Nodes",i + 1)
                headFeaturesTrain,headFeaturesTest,TargetsTrain,TargetsTest = forFinalEvaluation(BigGraph.Output(),ObjectiveList) 
                PrintAccuracy(headFeaturesTrain,headFeaturesTest,TargetsTrain,TargetsTest)

            print("\r\n")
            print(datetime.now().strftime("%Y-%m-%d %H:%M"),"\r\n")

class Logger():
    def __init__(self,file):
        self.terminal = sys.stdout
        self.log = open(file, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass  
    
def calculCorrelationMatrice(File,FeatureList):
    
    corrMatrice = []
    for feature in FeatureList:
        CorrList = []
        for feature1 in FeatureList:
            if feature == feature1:
                continue
            else:
                #one = feature.TrainValue ^ feature.NegTrainValue
                #vectCorr = (feature.TrainValue ^ feature1.TrainValue) ^ one
                Corr = feature.NbTrain - popcount(feature.TrainValue ^ feature1.TrainValue)
                list = [feature1,2 * abs((Corr/feature.NbTrain) - 0.5)]
                CorrList.append(list)
        corrMatrice.append(dict(CorrList))
    correlationMatrice = [sorted(x.items(), key = lambda x: x[1], reverse=True) for x in corrMatrice ]
    
    f = open(File + ".dump", "wb")
    pickle.dump(correlationMatrice,f)
    f.close()
    #return correlationMatrice
    

##############################################################################
##############################################################################
##############################################################################
# End of function and procedures and beganing of test execusion.
if __name__ == "__main__":	
    pool = multiprocessing.Pool(4)
    #sys.stdout = open('test_bagging_with norm', 'w')
    print("--- ",end="")
    print("START ---")
            
    
    win = graphics.GraphWin("TEST",1000,600)
    Line = graphics.Line(graphics.Point(1,100), graphics.Point(1000, 100))
    Line.setFill('blue')
    Line.draw(win)
    posx = [10]


    File = "MNIST-01234-56789_3600-3000-400"
    File1 = "CorrelationMatrice_MNIST-01234-56789_3600-3000-400"

    #File = "MNIST-01234-56789_46000-10000-6000"
    #File1 = "CorrelationMatrice_MNIST-01234-56789_46000-10000-6000"

    #File = "MNIST-01234-56789_9000-1000-1000"
    #File1 = "CorrelationMatrice_MNIST-01234-56789_9000-1000-1000"

    #File = "MNIST-01234-56789_800-1000-200"
    #File1 = "CorrelationMatrice_MNIST-01234-56789_800-1000-200"

    #File = "MNIST-MultiClass_46000-10000-6000"
    #File1 = "CorrelationMatrice_MNIST-MultiClass_46000-10000-6000"

    #File = "MNIST-MultiClass_9000-1000-1000"
    #File1 = "CorrelationMatrice_MNIST-MultiClass_9000-1000-1000"

    #File = "MNIST-MultiClass_800-1000-200"
    #File1 = "CorrelationMatrice_MNIST-MultiClass_800-1000-200"

    #File = "CIFAR-1bpc_45000-10000-5000"
    #File1 = "CorrelationMatrice_CIFAR-1bpc_45000-10000-5000"

    #File = "CIFAR-2bpc_45000-10000-5000"
    #File1 = "CorrelationMatrice_CIFAR-2bpc_45000-10000-5000"

    #File = "CIFAR-3bpc_45000-10000-5000"
    #File1 = "CorrelationMatrice_CIFAR-3bpc_45000-10000-5000"

    #File = "CONVEX_7500-50000-500"
    #File1 = "CorrelationMatrice_CONVEX_7500-50000-500"

    #File = "CONVEX_7500-4000-500"
    #File1 = "CorrelationMatrice_CONVEX_7500-4000-500"

    #File = "CIFAR-2CAT-FrAut-1bpc_10000-1000-1000"
    #File1 = "CorrelationMatrice_CIFAR-2CAT-FrAut-1bpc_10000-1000-1000"

    #File = "CIFAR-2CAT-FrAut-2bpc_10000-1000-1000"
    #File1 = "CorrelationMatrice_CIFAR-2CAT-FrAut-2bpc_10000-1000-1000"
        
    #File = "CIFAR-2CAT-FrAut-3bpc_10000-1000-1000"
    #File1 = "CorrelationMatrice_CIFAR-2CAT-FrAut-3bpc_10000-1000-1000"

    #File = "CIFAR-2CAT-DeeHor-1bpc_10000-1000-1000"
    #File1 = "CorrelationMatrice_CIFAR-2CAT-DeeHor-1bpc_10000-1000-1000"

    #File = "CIFAR-2CAT-DeeHor-2bpc_10000-1000-1000"
    #File1 = "CorrelationMatrice_CIFAR-2CAT-DeeHor-2bpc_10000-1000-1000"

    #File = "CIFAR-2CAT-DeeHor-3bpc_10000-1000-1000"
    #File1 = "CorrelationMatrice_CIFAR-2CAT-DeeHor-3bpc_10000-1000-1000"


    #GenDataMNIST_BW_10Cat(File, [0,1,2,3,4,5,6,7,8,9], Noise = 0.0, Type="basic", NbTrain = 800, NbTest = 1000, NbValid = 200)
    #GenDataMNIST_BW_2Cat(File,0,1, Noise = 0.0, Type = "convex", NbTrain = 7500, NbTest = 4000, NbValid = 500)

    #GenDataCIFAR_BW_10Cat(File,3,45000,10000,5000)
    #GenDataCIFAR_BW_2Cat(File,3,10000,1000,1000,4,7)
    
    #GenTwoCubesDataSet(File,"32x32 1 or two cubes, noise = 0.2", 5000, 10000, 5000, 0.2)
    
    #GenTwoNormal(File, "Normal mu0 = 10000, sigma0=100 mu1=10000, sigma1=500,16 element", 1000, 1000, 500, 16, 10000,100, 10000, 200)

    FileLog = "TestLog"

    sys.stdout = Logger(FileLog)
    
    Data = LoadDataSet(File + "T")
    #Data = LoadDataSet(File + "V")

    FeatureList = Data.Features
    Objective = Data.Objective
    #Data.BinaryAfficheImageRandom(4)
    
    #FeatureList = Data.Features
    #ObjectiveList = Data.Objectives
    #Data.AfficheImageRandom(3)

    #FeatureList = EliminateLowEntropyFeature(FeatureList_basic,0.3)
    
    #calculCorrelationMatrice(File1,FeatureList)
    f = open(File1+".dump","rb")
    CorrelationMatrice = pickle.load(f)


    BinaryGraph(6,[1,6,12,24,48],[1,6,12,24,48,96,192],48,FeatureList,Objective,0,10,0,10,0,10,0,10,783,CorrelationMatrice,hd = 2,Eps_top_1 = 0.02, Eps_bottom_1 = 0.02,Eps_top_2 = 0.02, Eps_bottom_2 = 0.02,Eps_top_3 = 0.02, Eps_bottom_3 = 0.02,BottomLevels_Prob = [1,6,12,24,48,96,192],TopLevels_Prob = [1,6,12,24,48],AllLevels_Prob = [1,6,12,24,48,48,288,576,1152,2304,4608,9216], choice = 0,bagging = 1)
    #BinaryGraph(4,[1,4,4,8],[1,4,8,16],8,FeatureList,Objective,0,10,0,10,0,10,0,10,783,CorrelationMatrice,hd = 2,Eps_top_1 = 0.02, Eps_bottom_1 = 0.02,Eps_top_2 = 0.02, Eps_bottom_2 = 0.02,Eps_top_3 = 0.02, Eps_bottom_3 = 0.02,BottomLevels_Prob = [1,4,8,16],TopLevels_Prob = [1,4,4,8],AllLevels_Prob = [1,4,4,8,16,8,32,64], choice = 0,bagging = 1)
    #MultiClassGraph(4,[1,10,10,20,40,80],[1,4,8,16,32,64,128],5,5,FeatureList,ObjectiveList,5,3,10,10,10,10,10,10,783,CorrelationMatrice,hd = 2,c = 3,Eps_top_1 = 0, Eps_bottom_1 = 0,Eps_top_2 = 0.05, Eps_bottom_2 = 0.1,Eps_top_3 = 0, Eps_bottom_3 = 0,BottomLevels_Prob = [1,4,8,16,32,64,128],TopLevels_Prob = [5,10,20,40,80], AllLevels_Prob = [10,10,20,40,80,32,128,256,512,1024,2048,4096], choice = 0, Target_sets_value = False,bagging = 1)
    #MultiClassGraph(6,[1,10,10,30,60],[1,6,12,24,48,96,192],20,16,FeatureList,ObjectiveList,0,3,10,10,783,CorrelationMatrice,c0 = 1,c1 = 0, c2 = 0,hd = 2,c = 3,Eps_top = 0, Eps_bottom = 0,BottomLevels_Prob = [1,6,12,24,48,96,192],TopopLevels_Prob = [5,10,30,60], AllLevels_Prob = [10,10,30,60,36,216,432,864,1728,3456,6912], choice = 0, Target_sets_value = False)
  


     #SVM

    #print("--------------------  Validation Process  -------------------------")
    #listOfBestScores = []
    #listOfScores = []
    #listOf_C_gamma = []
    #for Ce in range(-1,6):
    #    for gammae in range(0,8):
    #        C=10.0**(Ce)
    #        print("C=",C)
    #        gamma = 10.0**(-gammae)
    #        print("gamma=",gamma)
    #        clf=svm.SVC(C=C, gamma=gamma)
    #        clf.fit(Data.TrainData,Data.TrainTarget)
    #        print("End of training for SVM")
    #        print("svm train :",clf.score(Data.TrainData,Data.TrainTarget))
    #        print("svm validate :", clf.score(Data.TestData,Data.TestTarget))
    #        listOf_C_gamma.append([clf.score(Data.TestData,Data.TestTarget), C, gamma])
            
    #print("--------------------  Round 2  -------------------------")
    #listOfScores = []
    #Max_score = max(x[0] for x in listOf_C_gamma)
    #list = [x for x in listOf_C_gamma if x[0] == Max_score]
    #for x in list:
    #    listOfGamma =[]
    #    gammae = int(mth.log10(x[2]) - 1)
    #    gamma = 0
    #    listOfGamma = []
    #    for i in range(gammae,int(mth.log10(x[2]))+1):
    #        listOfGamma.append(10**(i))
    #        gamma = 0
    #        for _ in range(4):
    #            gamma = gamma + 2 * 10**(i)
    #            listOfGamma.append(round(gamma,8))
    #    del(listOfGamma[0])
    #    listOfC = []
    #    half = x[1]/2
    #    listOfC = [half,x[1],round(x[1]+half,8)]
    #    for C1 in listOfC:
    #        for gamma1 in listOfGamma:
    #            print("C=",C1)
    #            print("gamma=",gamma1)
    #            clf1=svm.SVC(C=C1, gamma=gamma1)
    #            clf1.fit(Data.TrainData,Data.TrainTarget)
    #            print("End of training for SVM")
    #            print("svm train :",clf1.score(Data.TrainData,Data.TrainTarget))
    #            print("svm validate :", clf1.score(Data.TestData,Data.TestTarget))
    #            listOfScores.append([clf1.score(Data.TestData,Data.TestTarget), C1, gamma1])
            
    #Max = max(j[0] for j in listOfScores)
    #print("Best validation SVM Score = ", Max)
    #listOfBestScores = [x for x in listOfScores if x[0] == Max]
    #BestHyperparameters = [listOfBestScores[0][1],listOfBestScores[0][2]]

    #print("-----------------------------     Test Process   --------------------------------------------")

    #Data1 = LoadDataSet(File + "T")
    #print("C=", BestHyperparameters[0])
    #print("gamma=", BestHyperparameters[1])
    #clf2=svm.SVC(C=BestHyperparameters[0], gamma=BestHyperparameters[1])
    #clf2.fit(Data1.TrainData,Data1.TrainTarget)
    #print("End of training for SVM")
    #print("svm train :",clf2.score(Data1.TrainData,Data1.TrainTarget))
    #print("svm test :", clf2.score(Data1.TestData,Data1.TestTarget))        
        
MultiClassBinaryClassifierAlain.py
Displaying MultiClassBinaryClassifierAlain.py.
