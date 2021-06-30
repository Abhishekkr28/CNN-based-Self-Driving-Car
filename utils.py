from keras import optimizers
# from Training import Steering
from imgaug.augmenters.flip import Fliplr
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import matplotlib.image as mpimg
from imgaug import augmenters as iaa
import cv2
import random
from keras.models import Sequential
from keras.layers import Convolution2D,Flatten,Dense
from keras.optimizers import Adam

 
# def getNames(file):
#     return file.split('\\')[-1] 
def importDatainfo(path):
    columns=['Center','Left','Right','Steering','Brake','Speed']
    data=pd.read_csv(os.path.join(path,'driving_log.csv') ,names=columns,index_col=False)
    
    # data['Center']=data['Center'].apply(getNames)
    # print(data.head())
    # print(data.shape[0])
    return data

def balanceData(data,dispaly=True):
    nBins=31
    samplesPerBin=5000
    hist,bins=np.histogram(data['Steering'],nBins)
    if dispaly:
        center=(bins[:-1]+bins[1:])*0.5
        # print(center)
        plt.bar(center,hist,width=0.06)
        plt.plot((-1,1),(samplesPerBin,samplesPerBin))
        plt.show()
    removeIndexList=[]
    for j in range(nBins):
        binDatalist=[]
        for i in range(len(data['Steering'])):
            if data['Steering'][i] >= bins[j] and data['Steering'][i]<=bins[j+1]:
                binDatalist.append(i)
        binDatalist=shuffle(binDatalist)
        binDatalist=binDatalist[samplesPerBin:]
        removeIndexList.extend(binDatalist)## append wont work
    # print('removed images ',len(removeIndexList))
    data.drop(data.index[removeIndexList],inplace=True)
    # print('Remaining Data ',len(data))
    if dispaly:
        hist,bins=np.histogram(data['Steering'],nBins)

        plt.bar(center,hist,width=0.06)
        plt.plot((-1,1),(samplesPerBin,samplesPerBin))
        plt.show()
    return data


def LoadData(data):
    imagePath=[]
    steering=[]

    X=data.values
    imagePath=X[:,0]
    steering=X[:,3] 
    return imagePath,steering  
    

def augmentImage(imgPath,steering):
    img=mpimg.imread(imgPath)
    if np.random.rand()<0.5:
        # PAN
        pan=iaa.Affine(translate_percent={'x':(-0.1,0.1),'y':(-0.1,0.1)})
        img=pan.augment_image(img)
    if np.random.rand()<0.5:
        # ZOOM
        zoom=iaa.Affine(scale=(1,1.2))
        img=zoom.augment_image(img)
    if np.random.rand()<0.5:
        # BRIGHTNESS
        brightness=iaa.Multiply((0.5,1.2)) # 0 dark 1 normal above 1 bright 
    if np.random.rand()<0.5:
        # Flip
        img=cv2.flip(img,1) #we need to flip the steering as well
        steering=-steering

    return img,steering


def preprocess(img):
    img=img[60:135,:,:]
    img=cv2.cvtColor(img,cv2.COLOR_RGB2YUV) # Lnaes become more visible
    img-cv2.GaussianBlur(img,(3,3),0)
    img=cv2.resize(img,(200,66))
    img=img/255 # normalization 0 to 1
    return img

# imgr=preprocess(mpimg.imread('temp.jpg'))
# plt.imshow(imgr)
# plt.show()

def batchGen(imagePath,steeringList,batchSize,Flag):
    while True:
        imgBatch=[]
        steeringBatch=[]
        for i in range(batchSize):
            index=random.randint(0,len(imagePath)-1)
            if Flag:
                img,steering=augmentImage(imagePath[index],steeringList[index])
            else:
                img=mpimg.imread(imagePath[index])
                steering=steeringList[index]    
            img=preprocess(img)
            imgBatch.append(img)
            steeringBatch.append(steering)

        yield (np.asarray(imgBatch),np.asarray(steeringBatch))


def Model():
    model=Sequential()
    model.add(Convolution2D(24,(5,5),strides=(2,2),input_shape=(66,200,3),activation='elu'))
    model.add(Convolution2D(36,(5,5),strides=(2,2),activation='elu'))
    model.add(Convolution2D(48,(5,5),strides=(2,2),activation='elu'))
    model.add(Convolution2D(64 ,(3,3),strides=(1,1),activation='elu'))
    model.add(Convolution2D(64,(3,3),strides=(1,1),activation='elu'))

    model.add(Flatten())
    model.add(Dense(100,activation='elu'))
    model.add(Dense(50,activation='elu'))
    model.add(Dense(10,activation='elu'))
    model.add(Dense(1))

    model.compile(Adam(lr=0.0001),loss='mse')
    return model