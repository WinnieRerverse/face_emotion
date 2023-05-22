# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 09:17:30 2017
"""

import tensorflow as tf
import numpy as np
import os
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split




data_file = open('data_48.pkl', 'rb')
label_file = open('label_48.pkl', 'rb')
data=pickle.load(data_file)
label=pickle.load(label_file)
data_array=np.array(data).reshape(len(data),data[0].shape[1])
label_array=np.array(label).reshape(len(data),1)
enc = OneHotEncoder()
enc.fit(label_array)  
label_array=enc.transform(label_array).toarray()


x_train, x_test, y_train, y_test = train_test_split(data_array, label_array)

batch=100
epochs=1000

def batch_iter(data, batch_size, num_epochs):
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, len(data))
            yield data[start_index:end_index]


in_units = 86#输入特征数 28*28
h1_units = 30#隐藏神经元个数

W1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.1))#生成截断正太分布
b1 = tf.Variable(tf.zeros(h1_units))

W2 = tf.Variable(tf.truncated_normal([h1_units, in_units], stddev=0.1))#生成截断正太分布
b2 = tf.Variable(tf.zeros(in_units))

W3 = tf.Variable(tf.zeros([in_units,7]))#隐藏层降维成7个表情特征
b3 = tf.Variable(tf.zeros([7]))

x = tf.placeholder(tf.float32, [None, in_units])
keep_prob = tf.placeholder(tf.float32)#dropout的概率


hidden1 = tf.nn.relu(tf.matmul(x,W1)+b1)
hidden1_drop = tf.nn.dropout(hidden1,keep_prob)


hidden2=tf.nn.relu(tf.matmul(hidden1_drop,W2)+b2)
hidden2_drop = tf.nn.dropout(hidden2,keep_prob)

y=tf.matmul(hidden2_drop,W3)+b3
y_=tf.placeholder(tf.int32,[None,7])#定义target输入
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y,labels=y_))
train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)#优化器


 #训练模型                                             
with tf.Session() as session:
    session.run(tf.initialize_all_variables())
    sum_loss=0
    batches = batch_iter(list(zip(x_train, y_train)),batch,epochs)
    for step, batch in enumerate(batches):
        x_batch, y_batch = zip(*batch)
        feed_dict = {x: x_batch, y_ : y_batch,keep_prob:1}
        _,loss = session.run([train_step, cross_entropy],feed_dict=feed_dict)
        sum_loss+=loss
        if step%1000==0:
            print("step {:d}, loss {:g}".format(step, sum_loss/step))
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    print("",session.run(accuracy,{x:x_test,y_:y_test,keep_prob:1.0}))







        




