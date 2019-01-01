# -*- coding: utf-8 -*-

import re
import os
from PIL import Image  
import cv2
from random import shuffle
import numpy as np

def saveImage(path,savedir):
    name=os.path.basename(path)
    img = cv2.imread(path)
    filename='{}{}'.format(savedir,name)
     #   print filename
    cv2.imwrite(filename,img)
    return  
def saveImage_rotate(path,savedir):
    name=os.path.basename(path)
    name=re.sub('.tif','',name)
    img=Image.open(path)
    img=img.rotate(180)  
    filename='{}{}{}{}'.format(savedir,os.sep,name,'_rotate.tif')
    img.save(filename)
    return  
def saveImage_tran_lift(path,savedir):
    name=os.path.basename(path)
    name=re.sub('tif','',name)
    img=Image.open(path)
    img = img.transpose(Image.FLIP_LEFT_RIGHT)   
    filename='{}{}{}{}'.format(savedir,os.sep,name,'_LEFT.tif')
    img.save(filename)
    return  
def saveImage_tran_lift_rotate(path,savedir):
    name=os.path.basename(path)
    name=re.sub('.tif','',name)
    img=Image.open(path)
    img = img.transpose(Image.FLIP_LEFT_RIGHT)   
    img=img.rotate(180) 
    filename='{}{}{}{}'.format(savedir,os.sep,name,'_LEFT_rotate.tif')
    img.save(filename)
    return  
def saveImage_tran_top(path,savedir):
    name=os.path.basename(path)
    name=re.sub('.tif','',name)
    img=Image.open(path)
    img = img.transpose(Image.FLIP_TOP_BOTTOM)   
    filename='{}{}{}{}'.format(savedir,os.sep,name,'_TOP.tif')
    img.save(filename)
    return  
def saveImage_tran_top_rotate(path,savedir):
    name=os.path.basename(path)
    name=re.sub('.tif','',name)
    img=Image.open(path)
    img = img.transpose(Image.FLIP_TOP_BOTTOM) 
    img=img.rotate(180) 
    filename='{}{}{}{}'.format(savedir,os.sep,name,'_TOP_rotate.tif')
    img.save(filename)
    return  
def saveImage_tran_top_left(path,savedir):
    name=os.path.basename(path)
    name=re.sub('.tif','',name)
    img=Image.open(path)
    img = img.transpose(Image.FLIP_TOP_BOTTOM) 
    img = img.transpose(Image.FLIP_LEFT_RIGHT)  
    filename='{}{}{}{}'.format(savedir,os.sep,name,'_TOP_LIFT.tif')
    img.save(filename)
    return  
def folderexist(savedir):
    if not os.path.exists(savedir):
#        print ("folder",savedir,'is not exists')
        os.makedirs(savedir)
#        print ('recreat over')    
    return
def getchildFolderName(filedir):
#filedir="/home/yinghuo/data/breast_cancer/breast cancer TBME2015 brazil/BreaKHis_v1/BreaKHis_v1/BreaKHis_v1/histology_slides/breast/benign/SOB/"
    subdir_list = [] 
    for dirpath, dirnames, files in os.walk(filedir):
         subdir_list=dirnames
         break
    return  subdir_list


def getAllImages(path):
    #f.endswith（）  限制文件类型
    #f.endswith('.jpg')|f.endswith('.png')  改成这句可以读取jpg/png格式的文件
    #注意 返回的是绝对路径
   return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.png')]   
def get_dataset_size(label,factor,dataset_dir):
    dataset_dir='{}{}{}'.format(dataset_dir,os.sep,label)
    num=0
    for subdir in getchildFolderName(dataset_dir):
        imagedir='{}{}{}'.format(dataset_dir,os.sep,subdir)
        imagedir='{}{}{}'.format(imagedir,os.sep,factor)
        if  os.path.exists(imagedir):
        
            for img in getAllImages(imagedir):
                num+=1
    size=num
    return size
def get_dataset_image(label,factor,dataset_dir):
    image_list=[]
    dataset_dir='{}{}{}'.format(dataset_dir,os.sep,label)
    for subdir in getchildFolderName(dataset_dir):
        imagedir='{}{}{}'.format(dataset_dir,os.sep,subdir)
        imagedir='{}{}{}'.format(imagedir,os.sep,factor)
        if os.path.exists(imagedir):
        
            for img in getAllImages(imagedir):
                 image_list.append([img,subdir])  
    
    return image_list

#size_0=625
size_1s=[1370,1437,1390,1232]
#test_dir='/media/yinghuo/data/data/breast/breast/data_2/patient_new/patient_image_2/test'
train_dir='/media/yinghuo/data/data/breast/breast/data_2/orl_data'
new_train_dir='/media/yinghuo/data/data/breast/breast/data_2/source'
folderexist(new_train_dir)

labels=range(0,2)
patients=[range(0,24),range(24,82)]
factors=['40X','100X','200X','400X']
for factor in factors:
    for label in labels:
        for patient in patients[label]:
            file_dir='{}{}{}{}{}{}{}{}'.format(train_dir,os.sep,label,os.sep,patient,os.sep,factor,os.sep)
            folderexist(file_dir)
            file_dir='{}{}{}{}{}{}{}{}'.format(new_train_dir,os.sep,label,os.sep,str(int(patient)),os.sep,factor,os.sep)
            folderexist(file_dir)
for i in range(0,4):
    size_1=size_1s[i]
    factor_new=factors[i]
    for factor in [factor_new]:
        for label in [0] :
           
                
                dataset_size=get_dataset_size(label,factor,train_dir)
                image_list=get_dataset_image(label,factor,train_dir)
                print ('The 0 data orl size is :',dataset_size)
                train_size_0=size_1-0
                print ('The needed 0 data orl size is :',train_size_0)
                if  int(dataset_size)<train_size_0:
                    new_image_list=image_list                
                    if int(dataset_size)+int(dataset_size)>=train_size_0:
                        shuffle(image_list)
                        new_image_list=image_list[0:train_size_0-(int(dataset_size))]
                    for img in new_image_list:
                        image_new_dir='{}{}{}{}{}{}{}{}'.format(new_train_dir,os.sep,label,os.sep,img[1],os.sep,factor,os.sep)
                        saveImage(img[0],image_new_dir)
#                        print ('save orl over!')
                        saveImage_rotate(img[0],image_new_dir)
#                        print ('save rotate over!')
                    new_dataset_size=get_dataset_size(label,factor,new_train_dir)
                    if  int(new_dataset_size)<train_size_0:
                        new_image_list=image_list 
                        if int(new_dataset_size)+int(dataset_size)>=train_size_0:
                            shuffle(image_list)
                            new_image_list=image_list[0:train_size_0-(int(new_dataset_size))]
                        for img in new_image_list:
                             image_new_dir='{}{}{}{}{}{}{}{}'.format(new_train_dir,os.sep,label,os.sep,img[1],os.sep,factor,os.sep)
                             saveImage_tran_lift(img[0],image_new_dir)
                        new_dataset_size=get_dataset_size(label,factor,new_train_dir)
                        if  int(new_dataset_size)<train_size_0:
                            new_image_list=image_list 
                            if int(new_dataset_size)+int(dataset_size)>=train_size_0:
                                shuffle(image_list)
                                new_image_list=image_list[0:train_size_0-(int(new_dataset_size))]
                            for img in new_image_list:
                                      image_new_dir='{}{}{}{}{}{}{}{}'.format(new_train_dir,os.sep,label,os.sep,img[1],os.sep,factor,os.sep)
                                      saveImage_tran_top(img[0],image_new_dir)
                            new_dataset_size=get_dataset_size(label,factor,new_train_dir)
                            if  int(new_dataset_size)<train_size_0:
                              new_image_list=image_list 
                              if int(new_dataset_size)+int(dataset_size)>=train_size_0:
                                  shuffle(image_list)
                                  new_image_list=image_list[0:train_size_0-(int(new_dataset_size))]
                              for img in new_image_list:
                                  image_new_dir='{}{}{}{}{}{}{}{}'.format(new_train_dir,os.sep,label,os.sep,img[1],os.sep,factor,os.sep)
                                  saveImage_tran_lift_rotate(img[0],image_new_dir)
                              new_dataset_size=get_dataset_size(label,factor,new_train_dir)
                              if  int(new_dataset_size)<train_size_0:
                                  new_image_list=image_list 
                                  if int(new_dataset_size)+int(dataset_size)>=train_size_0:
                                      shuffle(image_list)
                                      new_image_list=image_list[0:train_size_0-(int(new_dataset_size))]
                                  for img in new_image_list:
                                          image_new_dir='{}{}{}{}{}{}{}{}'.format(new_train_dir,os.sep,label,os.sep,img[1],os.sep,factor,os.sep)
                                          saveImage_tran_top_rotate(img[0],image_new_dir)
                                  new_dataset_size=get_dataset_size(label,factor,new_train_dir)
                                  if  int(new_dataset_size)<train_size_0:
                                      new_image_list=image_list 
                                      if int(new_dataset_size)+int(dataset_size)>=train_size_0:
                                          shuffle(image_list)
                                          new_image_list=image_list[0:train_size_0-(int(new_dataset_size))]
                                      for img in new_image_list:
                                              image_new_dir='{}{}{}{}{}{}{}{}'.format(new_train_dir,os.sep,label,os.sep,img[1],os.sep,factor,os.sep)
                                              saveImage_tran_top_left(img[0],image_new_dir)
    
                                     
                                                   
#    for factor in [factor_new]:
#        for label in [1] :
#                
#                dataset_size=get_dataset_size(label,factor,train_dir)
##                print (dataset_size)
#                train_size_0=size_1-get_dataset_size(label,factor,test_dir)
#                if  int(dataset_size)<=train_size_0:
#                    image_list=get_dataset_image(label,factor,train_dir)
#                    for img in image_list:
#                        image_new_dir='{}{}{}{}{}{}{}{}'.format(new_train_dir,os.sep,label,os.sep,img[1],os.sep,factor,os.sep)
#                        saveImage(img[0],image_new_dir)
##                        print ('save orl over!')
#                        
#                    shuffle(image_list)
#                    image_list=image_list[0:train_size_0-dataset_size]
#                    for img in image_list:
#                        image_new_dir='{}{}{}{}{}{}{}'.format(new_train_dir,os.sep,label,os.sep,img[1],os.sep,factor)
#                        
#                        saveImage_rotate(img[0],image_new_dir)
##                        print ('save rotate over!')
    print (factor_new)                
   
    dataset_size=get_dataset_size(0,factor_new,new_train_dir)
    print ('The 0 data final size is :',dataset_size)  
   