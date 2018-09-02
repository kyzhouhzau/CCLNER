#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:zhoukaiyin
@Time:2018/5/12
@Bi-LSTM and decode layer
"""
import sys
sys.path.append(r'Bilstm_NER')
import tensorflow as tf
import pickle
import os
import glob
from LSTM_Decode import Model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def train_epoch(inputss,labelss,drop_keep_rate,lis_len):
        sess, train_step, loss, acc, saver, merged, \
        write, result = model.LSTM_At(inputss,labelss,drop_keep_rate,lis_len,mode="train")
        return sess, train_step, loss, acc, saver, merged, write, result

def load_pre(pre_path):
    with open(pre_path,'r') as rf:
        for line in rf:
            word = line.split('\t')[0]
            yield word

def pre_result(base_path,re):
    id2w = open(base_path + "id2label.pkl", 'rb')

    id2w_ = pickle.load(id2w, encoding='bytes')

    sent = []
    for i in range(1,len(re)-1):

        try:
            tag = id2w_[re[i]]
            sent.append(tag)
        except KeyError:
            tag = id2w_[2]
            sent.append(tag)
    return sent

if __name__ == "__main__":
    base_path = "predict\\"
    path_data = "result.txt"
    files = glob.glob('model/*.index')[0]
    EPOCH = files.split('-')[-1].split('.')[0]
    model_Path = "model/model.ckpt-" + EPOCH
    model = Model(1, base_path, path_data)
    words_, labels_ = model.load_data()
    batchsize=1
    inputss = tf.placeholder(tf.int32, [batchsize, None], name="inputss")
    labelss = tf.placeholder(tf.int32, [batchsize, None], name="labelss")
    lis_len = tf.placeholder(tf.int32, [batchsize], name="truelen")
    drop_keep_rate = tf.placeholder(tf.float32, name="dropout_keep")
    sess, train_step, loss, acc, saver, merged, write, result=train_epoch(inputss,
                                                                          labelss,
                                                                          drop_keep_rate,
                                                                          lis_len)
    saver.restore(sess, model_Path)
    with open(base_path+"tag_result.txt",'w') as wf:
        for i in range(len(words_)):
            x = []
            x.append(words_[i])
            x, lis_len_x = model.pooling(x)
            # for x in words_:
            #     k = len(x)
            #     x =list(np.expand_dims(x,0))
            #     print(k)
            # sess.run(train_step, feed_dict={inputss: x,lis_len:lis_len_x,drop_keep_rate:1})
            re = sess.run(result, feed_dict={inputss: x,lis_len:lis_len_x,drop_keep_rate:1})
            sent=pre_result(base_path,re)
            for label in sent:
                wf.write(label+'\n')
            wf.write('\n')


