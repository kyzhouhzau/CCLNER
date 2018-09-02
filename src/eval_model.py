#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:zhoukaiyin
"""
import sys
from collections import defaultdict
import tensorflow as tf
import AEP_model
import time
import glob
from collections import deque
words_, labels_ = AEP_model.load_data()
num_classes = 38
learning_rate = 1e-4
num_units = 50
decode_units = 50
evaldata_size = AEP_model.traindata_size
#适用小样本数据
batchsize = round(len(words_) * 1)-round(len(words_) * evaldata_size)
# batchsize=14

inputss = tf.placeholder(tf.int32, [batchsize, None], name="inputss")
labelss = tf.placeholder(tf.int32, [batchsize, None], name="labelss")
lis_len = tf.placeholder(tf.int32, [batchsize], name="truelen")
drop_keep_rate = tf.placeholder(tf.float32, name="dropout_keep")
sess, train_step, loss, acc, saver, merged, write, result = AEP_model.LSTM_At(inputss,
                                                                                     labelss,
                                                                                     drop_keep_rate,
                                                                                     lis_len,
                                                                                     num_units,
                                                                                     num_classes,
                                                                                     decode_units,
                                                                                     learning_rate,
                                                                                     batchsize=batchsize)
# sess, train_step, loss, acc, saver, merged, write,result = AEP_model.indrnn_model(inputss, labelss, num_units, num_classes, decode_units, learning_rate,lis_len,batch_size=batchsize)
# sess, train_step, loss, acc, saver, merged, write, result = AEP_model.True_model(inputss,
#                                                                                    labelss,
#                                                                                    drop_keep_rate,
#                                                                                    lis_len,
#                                                                                    batchsize,
#                                                                                    num_units,
#                                                                                    num_classes,
#                                                                                    learning_rate,
#                                                                                    decode_units)

def eval_result(sess,drop_keep_rate,lis_len,result,inputss,start, end,batchsize,words_,labels_):
    datasize_t = round(len(words_) * start)
    start_t = datasize_t
    end_t = round(len(words_) * end)
    poch_t = int((end_t - start_t) / batchsize)
    di = defaultdict(list)
    for i in range(poch_t):
        x_t = words_[start_t:start_t + batchsize]
        label_need_t = labels_[start_t:start_t + batchsize]
        x_t,lis_len_x = AEP_model.pooling(x_t)
        label_t,lis_len_x = AEP_model.pooling(label_need_t)
        # y = sess.run(result, feed_dict={inputss: x_t,drop_keep_rate:1.0})
        y = sess.run(result, feed_dict={inputss: x_t,lis_len:lis_len_x,drop_keep_rate:1.0})
        # y = sess.run(result, feed_dict={inputss: x_t,lis_len:lis_len_x})
        if i == 0:
            print("Waiting we are evaling the model")
        pre_label, need_label = AEP_model.get_sequence(y, label_t)
        for label, P, R, F in AEP_model.caculate_f(pre_label, need_label):
            di[label+"_P"].append(P)
            di[label+"_F"].append(F)
            di[label+"_R"].append(R)
        start_t = start_t + batchsize

    for i,lis in di.items():
        label=i.split('_')[0]
        T=i.split('_')[1]
        if T=="F":
            F=sum(di[i])
            print("{}:\t\tF:{}\t".format(label,F/poch_t))

def evaluate():
    saver.restore(sess, model_Path)
    eval_result(sess, drop_keep_rate,lis_len, result,inputss, evaldata_size, 1, batchsize, words_, labels_)

if __name__=='__main__':
    count = 0
    flag=True
    epoch_name=deque([0,1,2],maxlen=3)
    while flag:
        try:
            files = glob.glob('../model/*.index')[0]

            EPOCH = files.split('-')[-1].split('.')[0]
            model_Path = "../model/model.ckpt-" + EPOCH
            epoch_name.append(EPOCH)
            if epoch_name[-1]==epoch_name[-2]:
                pass
            else:
                print(">>>>>>>>>>>>>>Test:{}\tEPOCH:{}>>>>>>>>>>>".format(count,EPOCH))
                # evaluate()
                saver.restore(sess, model_Path)
                eval_result(sess, drop_keep_rate, lis_len, result, inputss, 0.7, 1, batchsize, words_, labels_)
                # time.sleep(5)
                count+=1
        except IndexError:
            pass


