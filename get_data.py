import os
import glob
import numpy as np
import pandas as pd
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.pyplot as plt
#import scipy.stats as stats
#tau, p_value = stats.kendalltau(x,y)
width =height = 224

train_data_path = "/media/yinghuo/data/data/breastpathq/datasets/new_train"

#train_data_path = "/media/yinghuo/data/data/breastpathq/datasets/train"
validation_data_path = "/media/yinghuo/data/data/breastpathq/datasets/validation"
lable_path = "/media/yinghuo/data/data/breastpathq/datasets"

def read_image(path_tra,path_val):
    print("======Loading data======")
    info = pd.read_csv("train_labels.csv")
    labels = np.array(info['y'].tolist())
    labels = labels.reshape(labels.shape[0],1)
    image_train = []
    image_validation = []
    name = glob.glob(os.path.join(path_tra, '*.tif'))
    name_val = glob.glob(os.path.join(path_val, '*.tif'))
    name.sort(key=lambda x:x[:-4])
    name_val.sort(key=lambda x:x[:-4])
#    print(name[:12])
#    print(name_val[:12])
    for k in range(len(name)):
        img = imread(name[k])
        img = imresize(img,(width,height))
        image_train.append(img)
    print("======train data Load finished======")
    for j in range(len(name_val)):
        img2 = imread(name_val[j])
        img2 = imresize(img2,(width,height))
        image_validation.append(img2)

    print("======validation data Load finished======")
    info = pd.read_csv("val_labels.csv")
    val_labels = np.array(info['y'].tolist())
    val_labels = val_labels.reshape(val_labels.shape[0],1)
    
    return np.array(image_train), labels, np.array(image_validation),val_labels
    
#==============================================================================
# if __name__=="__main__":
#     train_x, train_y, test_x,test_y = read_image(train_data_path,validation_data_path)
#==============================================================================

def read_image_qi(path_tra,path_val):
    print("======Loading data======")
    info = pd.read_csv(lable_path+os.sep+"new_train.csv")
    labels = np.array(info['y/n'].tolist())
    labels = labels.reshape(labels.shape[0],1)
    slides = np.array(info['slide+rid'].tolist())
#    rids = np.array(info['rid'].tolist())
    image_train = []
    
#    print(name[:12])
#    print(name_val[:12])
    for k in range(len(labels)):
        img = imread(path_tra+os.sep+str(slides[k])+".tif")
        img = imresize(img,(width,height))
        image_train.append(img)
    print("======train data Load finished======")
    
    image_validation=[]
    info_val = pd.read_csv("val_labels.csv")
    val_labels = np.array(info_val['y'].tolist())
    val_labels = val_labels.reshape(val_labels.shape[0],1)
    val_slides = np.array(info_val['slide'].tolist())
    val_rids = np.array(info_val['rid'].tolist())
    for j in range(len(val_labels)):
        img2 = imread(path_val+os.sep+str(int(val_slides[j]))+"_"+str(val_rids[j])+".tif")
        img2 = imresize(img2,(width,height))
        image_validation.append(img2)

    print("======validation data Load finished======")
    
    return np.array(image_train), labels, np.array(image_validation),val_labels

#if __name__=="__main__":
#     train_x, train_y, test_x,test_y = read_image_qi(train_data_path,validation_data_path)