# -*- coding: utf-8 -*-
import os
#import scipy.stats as stats
from get_data import *
from net.vgg16  import  vgg16

import tensorflow as tf
import numpy as np
import get_data as gd
import datasets_None_prediction_probability as out
from random import shuffle
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
weight_decay = 0.0001
momentum = 0.9
CUDA_poi=0.9
init_learning_rate = 0.0001
if_train=True
batch_size = 1#18
#iteration = 5000
iteration = 14364
TEST_SET_SIZE=185
# 128 * 391 ~ 50,000
enlarge_factor=100
total_epochs = 1000

class_num = 1
image_size = 224
# =============================================================================
# image_size = 224#256-Senet,227-alexnet
# =============================================================================
img_channels = 3
#p_list = []
#train_x, train_y, test_x, test_y = prepare_data()
#train_x, test_x = color_preprocessing(train_x, test_x)
train_x, train_y, test_x,test_y = gd.read_image_qi(train_data_path,validation_data_path)

# image_size = 32, img_channels = 3, class_num = 10 in cifar10

x = tf.placeholder(tf.float32, shape=[None, image_size, image_size, img_channels])
label = tf.placeholder(tf.float32, shape=[None, class_num])

training_flag = tf.placeholder(tf.bool)

enlarge_rate = tf.placeholder(tf.float32, name='enlarge_rate')
learning_rate = tf.placeholder(tf.float32, name='learning_rate')

model = vgg16(x,1)
# Link variable to model output
logits = model.probs

# =============================================================================
# logits = densenet.densenet_inference(x,training_flag, dropout_prob=0.7)
# =============================================================================
cost = tf.reduce_mean(tf.square(logits - label*enlarge_rate))
l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
optimizer =tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(cost + l2_loss * weight_decay)

correct_prediction = tf.equal(logits,label)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver(tf.global_variables())
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=CUDA_poi)  

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    ckpt = tf.train.get_checkpoint_state('./alex_model')
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())
#        vgg16.load_weights(model,weight_file='./vgg16_weights.npz',sess=sess)

    summary_writer = tf.summary.FileWriter('./alex_logs', sess.graph)

    epoch_learning_rate = init_learning_rate
    best_auc=0
    shuffle(train_x)
    train_best_auc=0
    for epoch in range(1, total_epochs + 1):
         train_label=[]
         real_label=[]
         if epoch % 50 == 0 :
             epoch_learning_rate = epoch_learning_rate / 2
 
         pre_index = 0
         train_acc = 0.0
         train_loss = 0.0
 
         for step in range(1, iteration + 1):
             if pre_index + batch_size < 14364:
                 batch_x = train_x[pre_index: pre_index + batch_size]
                 batch_y = train_y[pre_index: pre_index + batch_size]
             else:
                 batch_x = train_x[pre_index:]
                 batch_y = train_y[pre_index:]

             train_feed_dict = {
             x: batch_x,
             label: batch_y,
             learning_rate: epoch_learning_rate,
             training_flag: True,
             enlarge_rate:enlarge_factor
             }
 
             _, batch_loss, b_scores = sess.run([train, cost, logits], feed_dict=train_feed_dict)
             batch_acc = accuracy.eval(feed_dict=train_feed_dict)
             if b_scores[0][0]/enlarge_factor>1:
                    b_scores[0][0]=1*enlarge_factor
             elif b_scores[0][0]/enlarge_factor<0:
                b_scores[0][0]=0*enlarge_factor
             train_label.append(round(b_scores[0][0]/enlarge_factor,3))
             real_label.append(round(batch_y[0][0],3))
#             if step % 5 == 0:
#                  print("iteration: %d/%d  batch_loss: %.4f  scores: %.3f" %(step,iteration,batch_loss,p_scores))
             train_loss += batch_loss
             train_acc += batch_acc
             pre_index += batch_size
  

         train_loss /= iteration # average loss
         train_acc /= iteration # average accuracy
        
         train_summary = tf.Summary(value=[tf.Summary.Value(tag='train_loss', simple_value=train_loss),
                                           tf.Summary.Value(tag='train_accuracy', simple_value=train_acc)])

         summary_writer.add_summary(summary=train_summary, global_step=epoch)
         summary_writer.flush()
         train_auc=out.predprob(train_label,real_label)
         line = "epoch: %d/%d, train_loss: %.4f,train_acc:%.4f,train_auc:%.4f, \n" % (
         epoch, total_epochs, train_loss,train_acc,train_auc)
         print(line)
         with open('logs.txt', 'a') as f:
              f.write(line)
         if train_auc >train_best_auc:
                 train_best_auc=train_auc
                 saver.save(sess=sess, save_path='./alex_model/Inception_v2.ckpt')
                 print ("best train result:",train_best_auc)
         else:
                 print ("best train result:",train_best_auc)
         
         if epoch%1==0:
             output_label=[]
             pre_index = 0
             for i in range(int(TEST_SET_SIZE / batch_size)):
                if pre_index + batch_size < TEST_SET_SIZE:
                     batch_x = test_x[pre_index: pre_index + batch_size]
                     batch_y = test_y[pre_index: pre_index + batch_size]
                else:
                     batch_x = test_x[pre_index:]
                     batch_y = test_y[pre_index:]
                test_feed_dict = {
                 x: batch_x,
                 label: batch_y,
                 learning_rate: epoch_learning_rate,
                 training_flag: False,
                 enlarge_rate:enlarge_factor
                 }
 
                _, batch_loss, p_scores = sess.run([train, cost, logits], feed_dict=test_feed_dict)
                if p_scores[0][0]/enlarge_factor>1:
                    p_scores[0][0]=1*enlarge_factor
                elif p_scores[0][0]/enlarge_factor<0:
                    p_scores[0][0]=0*enlarge_factor
                output_label.append(round(p_scores[0][0]/enlarge_factor,3))
                line="output result:%.4f,real_label:%.4f"%(round(p_scores[0][0],3),batch_y[0][0])
                print(line)
                pre_index += batch_size
#                print (pre_index)
             auc=out.predprob(test_y,output_label)
             print ("val result:",auc)
             if auc >best_auc:
                 best_auc=auc
                 saver.save(sess=sess, save_path='./alex_val_model/Inception_v2.ckpt')
                 print ("best val result:",best_auc)
             else:
                 print ("best val result:",best_auc)