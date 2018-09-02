#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:zhoukaiyin
@Time:2018/5/12
@AEP_model.py
"""
import os
import sys
import glob
import pickle
import numpy as np
import tensorflow as tf
from collections import deque
from collections import Iterable
from collections import defaultdict
from tensorflow.contrib import rnn
from rnn_cell.My_LSTM import NLSTM, TLSTM
from rnn_cell.ind_rnn_cell import IndRNNCell
from tensorflow.python.layers.core import Dense
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if len(sys.argv)<2:
    print("Using:python AEP_model.py [LSTM_attention/LSTM_CRF/indrnn]")
    exit()
ty = sys.argv[1]


class Model(object):
    def __init__(self,batchsize,base_path,path_data):
###############超参数############
        self.Train = True
        if self.Train:
            self.batchsize = batchsize
        else:
            self.batchsize = batchsize
        # embed_dim = 200
        self.num_classes = 6
        self.learning_rate = 1e-4
        self.num_units = 100
        self.decode_units = 100
        self.L2_penalty = 1e-4
        self.traindata_size=0.7
        self.model_type = ty
        # self.model_type = "indrnn"
        self.base_path = base_path
        self.path_data = path_data
        self.model_Path = "../model/model.ckpt"
#############超参数#############
    def test_pkl(self,path):
        rf = open(path, 'rb')
        result = pickle.load(rf, encoding='bytes')
        rf.close()
        return len(result)
        # print(np.array(result))

    def load_data(self):
        label_lis = []
        sentence_lis = []
        li_w = []
        li_label = []
        words = []
        labels = []
        rf = open(self.base_path+self.path_data, 'r')
        w2id = open(self.base_path+"w2id.pkl", 'rb')
        w2id_ = pickle.load(w2id, encoding='bytes')
        label2id = open(self.base_path+"label2id.pkl", 'rb')
        label2id_ = pickle.load(label2id, encoding='bytes')
        li_w.append('<start>')
        li_label.append('O')
        for line in rf:
            word = line.split('\t')[0]
            label = line.strip().split('\t')[-1]
            if len(line) > 1:
                li_w.append(word)
                li_label.append(label)
            else:
                li_w.append('<end>')
                li_label.append('O')
                if len(li_w) <= 0:
                    pass
                else:
                    label_lis.append(li_label)
                    sentence_lis.append(li_w)
                li_w = []
                li_label = []
                li_w.append('<start>')
                li_label.append('O')

        for sent in sentence_lis:
            a = []
            for w in sent:
                a.append(w2id_[w])
            words.append(a)
        for lab in label_lis:
            b = []
            for l in lab:
                b.append(label2id_[l])
            labels.append(b)
        print("len(words){}\tlen(labels){}".format(len(words), len(labels)))
        rf.close()
        w2id.close()
        label2id.close()
        return words, labels


    def pooling(self,lis):
        lis_len = []
        result = []
        for sentence in lis:
            lis_len.append(len(sentence))
        max_len = max(lis_len)
        for sent in lis:
            if len(sent) <= max_len:
                leng_line = sent + [0.0] * (max_len - len(sent))
            else:
                leng_line = sent[:max_len]
            result.append(leng_line)
        return result ,lis_len


    def load_embed(self,path):
        rf = open(path, 'rb')
        word_embed = pickle.load(rf, encoding='bytes')

        # 注意这里的np.float32
        return np.array(word_embed, np.float32)


    def weight_init(self,shape):
        # [minval,maxval]之间的均匀分布。
        initial = tf.random_uniform(shape, minval=-np.sqrt(5) * np.sqrt(1.0 / shape[0]),
                                    maxval=np.sqrt(5) * np.sqrt(1.0 / shape[0]))
        return tf.Variable(initial, trainable=True,dtype=tf.float32)


    def bias_init(self,shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial, trainable=True,name='b')


    def shufflelist(self,words, labels):
        ri = np.random.permutation(len(words))  # 生成一个随机排列
        wordss = [words[i] for i in ri]
        labelss = [labels[i] for i in ri]
        return wordss, labelss


    def flatten(self,items, ignore_types=(bytes)):
        for x in items:
            if isinstance(x, Iterable) and not isinstance(x, ignore_types):
                yield from self.flatten(x)
            else:
                yield x


    def get_sequence(self,y, label_need):
        # 获取序列
        pre_label = []
        need_label = []
        for l in self.flatten(y):
            pre_label.append(l)
        for w in self.flatten(label_need):
            try:
                need_label.append(w)
            except KeyError:
                need_label.append(0)
        return pre_label, need_label


    def caculate_f(self,base_path,pre_label, need_label):
        # 计算F值
        rf = open(base_path+"id2label.pkl", 'rb')
        labelDir = pickle.load(rf, encoding='bytes')
        labelDir[0] = "0"
        for i in range(1, len(labelDir)):
            label = labelDir[i]
            TP = 0.0000001
            FN = 0.0000001
            FP = 0.0000001
            TN = 0.0000001
            for k, nlabel in enumerate(need_label):
                if labelDir[nlabel] == label and nlabel == pre_label[k]:
                    TP += 1
                elif labelDir[nlabel] == label and nlabel != pre_label[k]:
                    FP += 1
                elif labelDir[nlabel] != label and nlabel == pre_label[k]:
                    TN += 1
                elif labelDir[nlabel] != label and nlabel != pre_label[k]:
                    FN += 1
            P = TP / (TP + FP)
            R = TP / (TP + FN)
            F = 2 * P * R / (P + R)
            yield label, P, R, F


    def eval_result(self,sess,lis_len,result,inputss,start, end,batchsize,test_x,test_y):

        words_,labels_=test_x, test_y
        # batchsize = round(len(words_)/fl)
        datasize_t = round(len(words_) * start)
        start_t = datasize_t
        end_t = round(len(words_) * end)
        poch_t = int((end_t - start_t) / batchsize)
        di = defaultdict(list)
        for i in range(poch_t):
            x_t = words_[start_t:start_t + batchsize]
            label_need_t = labels_[start_t:start_t + batchsize]
            x_t,lis_len_x = self.pooling(x_t)
            label_t,lis_len_x = self.pooling(label_need_t)
            # y = sess.run(result, feed_dict={inputss: x_t,drop_keep_rate:1.0})
            y = sess.run(result, feed_dict={inputss: x_t,lis_len:lis_len_x,drop_keep_rate:1.0})
            # y = sess.run(result, feed_dict={inputss: x_t,lis_len:lis_len_x})
            if i == 0:
                print("Waiting we are evaling the model")
            pre_label, need_label = self.get_sequence(y, label_t)
            for label, P, R, F in self.caculate_f(self.base_path,pre_label, need_label):

                di[label+"/P"].append(P)
                di[label+"/F"].append(F)
                di[label+"/R"].append(R)
            start_t = start_t + batchsize
        FF={}
        PP={}
        RR={}
        for i,lis in di.items():
            label=i.split('/')[0]
            T=i.split('/')[1]
            if T=="F":
                F=sum(di[i])
                avg = F/poch_t
                avg_f = round(avg,5)

                print("{}:\t\tF:{}\t".format(label,avg_f))
                FF[label]=avg_f
            if T=="P":
                P=sum(di[i])
                avg = P/poch_t
                avg_f = round(avg,5)
                PP[label] = avg_f
                print("{}:\t\tP:{}\t".format(label,avg_f))
            if T=="R":
                R=sum(di[i])
                avg = R/poch_t
                avg_f = round(avg,5)
                RR[label] = avg_f
                print("{}:\t\tR:{}\t".format(label,avg_f))
        return FF,PP,RR

    def split(self,data_x, n=5):
        assert isinstance(n, int), "n is not integer:\t %s"%(n)
        size = len(data_x)
        # assert size > n*2, "data size is too small"
        step = size//n
        index = np.r_[0:size]
        np.random.shuffle(index)
        indices = [index[i*step:(i+1)*step] for i in range(n)]
        nindices = [np.array(list(set(index) - set(index_i))) \
                    for index_i in indices]
        for i in nindices:
            np.random.shuffle(i)

        return zip(indices, nindices)

# Indrnn_model
# Title:Independently Recurrent Neural Network (IndRNN): Building A Longer and Deeper RNN
    def indrnn_model(self,inputss, labelss,lis_len,crf=False):
        ####
        ebend = self.load_embed(self.base_path+"w2embed.pkl")
        ebend_ = tf.Variable(ebend, name="ebd_weight", trainable=False, dtype=tf.float32)
        ebmeding = tf.nn.embedding_lookup(ebend_, inputss)
        TIME_STEPS = 20
        RECURRENT_MAX = pow(2, 1 / TIME_STEPS)
        input_init = tf.random_uniform_initializer(-0.001, 0.001)
        first_input_init = tf.random_uniform_initializer(-RECURRENT_MAX,RECURRENT_MAX)
        one_cell = IndRNNCell(self.num_units,input_kernel_initializer=input_init,recurrent_kernel_initializer=first_input_init)
        two_cell = IndRNNCell(self.num_units)
        cell = tf.nn.rnn_cell.MultiRNNCell([one_cell,two_cell])
        init_fw = cell.zero_state(self.batchsize, dtype=tf.float32)
        rnn1, state = tf.nn.dynamic_rnn(cell, ebmeding,
                                          sequence_length=lis_len,
                                          initial_state=init_fw,
                                          dtype=tf.float32)
        decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.num_units)

        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(50, rnn1,memory_sequence_length=lis_len)
        decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism, attention_layer_size=50)
        decoder_state = decoder_cell.zero_state(self.batchsize, tf.float32)
        helper = tf.contrib.seq2seq.TrainingHelper(inputs=ebmeding,
                                                   sequence_length=lis_len)

        decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, decoder_state)

        rnn_, _, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder=decoder,
            output_time_major=False)
        rnn2 = tf.reshape(rnn_[0], [-1, self.decode_units])
        W0 = self.weight_init([self.decode_units, self.num_classes])
        b0 = self.bias_init([self.num_classes])
        output = tf.matmul(rnn2, W0) + b0
        unary_scores = tf.reshape(output, [self.batchsize, -1, self.num_classes])
        del ebend
        # tv = tf.trainable_variables()  # 得到所有可以训练的参数，即所有trainable=True 的t
        # f.Variable/tf.get_variable
        # regularization_cost = L2_penalty * tf.reduce_sum([tf.nn.l2_loss(v) for v in tv])  # 0.001是lambda超参数
        with tf.name_scope("loss"):
            if crf:
                log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(unary_scores, labelss,sequence_lengths=lis_len) # need to evaluate it for decoding
                # loss = tf.reduce_mean(-log_likelihood)+regularization_cost
                loss = tf.reduce_mean(-log_likelihood)
                decode_tags, best_score = tf.contrib.crf.crf_decode(unary_scores, trans_params, lis_len)
                label_equal = tf.equal(tf.reshape(decode_tags,[-1]), tf.reshape(labelss, [-1]))
                acc = tf.reduce_mean(tf.cast(label_equal, tf.float32))
                result = tf.reshape(decode_tags,[-1])
            else:
                loss = tf.reduce_mean(
                    # tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(labelss, [-1]), logits=output))+regularization_cost
                    tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(labelss, [-1]), logits=output))
                label_equal = tf.equal(tf.cast(tf.argmax(output, 1), tf.int32), tf.reshape(labelss, [-1]))
                acc = tf.reduce_mean(tf.cast(label_equal, tf.float32))
                result = tf.argmax(output, 1)
            tf.summary.scalar("loss", loss)
        global_step = tf.Variable(0)
        # learning_r = tf.train.exponential_decay(
        #     learning_rate, global_step, 1000, 0.9, staircase=True)
        with tf.name_scope('train'):
            optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(loss, global_step=global_step)
        saver = tf.train.Saver()
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        merged = tf.summary.merge_all()
        write = tf.summary.FileWriter(r'logs', sess.graph)
        return sess, optimizer, loss, acc, saver, merged, write, result

    # Title:Joint Extraction of Entities and Relations Based on a Novel Tagging Scheme
    #LSTM+LSTM_Variable of LSTM_At
    def LSTM_At(self,inputss, labelss,drop_keep_rate,lis_len,mode="train",lstm_at=True,crf=False):
        ####
        global helper, rnn2
        ebend = self.load_embed(self.base_path+"w2embed.pkl")
        # ebend_ = tf.Variable(ebend, name="ebd_weight", trainable=False, dtype=tf.float32)
        ebmeding = tf.nn.embedding_lookup(ebend, inputss)
        # f_cell = NLSTM(num_units, bias=1.0)#这个NLSTM是自己根据tensorflow公式写的，可改动行大，经过测试期效果和封装的LSTMcell差不多
        # b_cell = NLSTM(num_units, bias=1.0)
        f_cell = rnn.LSTMCell(self.num_units,forget_bias=1.0)
        b_cell = rnn.LSTMCell(self.num_units,forget_bias=1.0)

        lstm_cell_fw = tf.nn.rnn_cell.DropoutWrapper(f_cell, output_keep_prob=(drop_keep_rate))
        lstm_cell_bw = tf.nn.rnn_cell.DropoutWrapper(b_cell, output_keep_prob=(drop_keep_rate))
        init_fw = lstm_cell_fw.zero_state(self.batchsize, dtype=tf.float32)
        inti_bw = lstm_cell_bw.zero_state(self.batchsize, dtype=tf.float32)
        # encoder_outputs, encoder_state = tf.nn.dynamic_rnn(lstm_cell_fw,  ebmeding,initial_state=init_fw )
        (rnn0, output_states) = tf.nn.bidirectional_dynamic_rnn(lstm_cell_fw,lstm_cell_bw, ebmeding,
                                                                sequence_length=lis_len,
                                                                initial_state_fw=init_fw,
                                                                initial_state_bw=inti_bw,
                                                                dtype=tf.float32)
        output_fw, output_bw = rnn0
        rnn1 = tf.concat(values=[output_fw, output_bw], axis=2)
        deco_r = tf.cast(rnn1,tf.float32)
        if lstm_at:
            s_cell = TLSTM(self.num_units, bias=1.0)
            init_s = s_cell.zero_state(self.batchsize, dtype=tf.float32)
            output, state = tf.nn.dynamic_rnn(s_cell, rnn1,
                                              initial_state=init_s,
                                              dtype=tf.float32)
            rnn2 = tf.reshape(output, [-1, self.decode_units])
        else:
            #decoderlayer

            # decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.num_units)
            decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(100)
            print(rnn1)
            # attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(50, rnn1,memory_sequence_length=lis_len)
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(100, rnn1,memory_sequence_length=lis_len)
            # decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell , attention_mechanism, attention_layer_size=50)
            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell , attention_mechanism, attention_layer_size=200)
            decoder_state = decoder_cell.zero_state(self.batchsize, tf.float32)
            if mode == "train":
                helper = tf.contrib.seq2seq.TrainingHelper(inputs=deco_r,
                                                        sequence_length=lis_len,
                                                        time_major=False, name='training_helper')

            elif mode == "infer":
                num=self.test_pkl(self.base_path + "/id2w.pkl") + 1
                # start_tokens = tf.tile(tf.constant([num], dtype=tf.int32), [self.batchsize],
                #                        name='start_token')
                # start_tokens = tf.fill([self.batchsize], num)
                # print(start_tokens)
                # end_tokens =tf.nn.embedding_lookup=(ebend,[num+1])
                end_tokens =num+1
                start_token=tf.tile(tf.constant([num], dtype=tf.int32), [self.batchsize], name='start_token')

                helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                    embedding=deco_r,
                    start_tokens=start_token,
                    end_token=end_tokens)
            decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, decoder_state)
            print("############################")
            rnn_, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=decoder, output_time_major=False)
            print("############################")
            rnn2 = tf.reshape(rnn_[0], [-1, self.decode_units])
            # rnn2 = tf.reshape(rnn_[0], [-1, self.decode_units*2])
        # 第三层
        del ebend
        # W0 = self.weight_init([self.decode_units, self.num_classes])
        W0 = self.weight_init([self.decode_units, self.num_classes])
        b0 = self.bias_init([self.num_classes])
        output = tf.matmul(rnn2 ,W0) + b0
        unary_scores = tf.reshape(output, [self.batchsize, -1, self.num_classes])
        # with tf.name_scope("accuracy"):
        #     label_equal = tf.equal(tf.cast(tf.argmax(output, 1), tf.int32), tf.reshape(labelss, [-1]))
            # acc = tf.reduce_mean(tf.cast(label_equal, tf.float32))
            # result = tf.argmax(output, 1)
        with tf.name_scope("loss"):
            if crf:
                log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(unary_scores, labelss,sequence_lengths=lis_len) # need to evaluate it for decoding
                # loss = tf.reduce_mean(-log_likelihood)+regularization_cost
                loss = tf.reduce_mean(-log_likelihood)
                decode_tags, best_score = tf.contrib.crf.crf_decode(unary_scores, trans_params, lis_len)
                label_equal = tf.equal(tf.reshape(decode_tags,[-1]), tf.reshape(labelss, [-1]))
                acc = tf.reduce_mean(tf.cast(label_equal, tf.float32))
                result = tf.reshape(decode_tags,[-1])
            else:
                loss = tf.reduce_mean(
                    # tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(labelss, [-1]), logits=output))+regularization_cost
                    tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(labelss, [-1]), logits=output))
                label_equal = tf.equal(tf.cast(tf.argmax(output, 1), tf.int32), tf.reshape(labelss, [-1]))
                acc = tf.reduce_mean(tf.cast(label_equal, tf.float32))
                result = tf.argmax(output, 1)
            tf.summary.scalar("loss", loss)
        # learning_r = tf.train.exponential_decay(
        #     learning_rate, global_step, 1500, 0.9, staircase=True)
        with tf.name_scope('train'):
            optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(loss)
        saver = tf.train.Saver()
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        merged = tf.summary.merge_all()
        write = tf.summary.FileWriter(r'logs', sess.graph)
        return sess, optimizer, loss, acc, saver, merged, write, result

    # tensorflow LSTM+CRF kernel
    #if crf=True we use crf to calculate best sequence.
    #if crf=FALSE we just use bi-lstm
    #LSTM+CRF
    def LSTM_CRF(self,inputss, labelss,drop_keep_rate, lis_len,crf=True):
        ####
        ebend = self.load_embed(self.base_path+"/w2embed.pkl")
        ebend_ = tf.Variable(ebend, name="ebd_weight", trainable=False, dtype=tf.float32)
        ebmeding = tf.nn.embedding_lookup(ebend_, inputss)
        del ebend
        rnn_fcell = rnn.LSTMCell(num_units=self.num_units, forget_bias=1.0)
        rnn_bcell = rnn.LSTMCell(num_units=self.num_units, forget_bias=1.0)
        lstm_cell_fw = tf.nn.rnn_cell.DropoutWrapper(rnn_fcell, output_keep_prob=(drop_keep_rate))
        lstm_cell_bw = tf.nn.rnn_cell.DropoutWrapper(rnn_bcell, output_keep_prob=(drop_keep_rate))
        inti_bw = lstm_cell_fw.zero_state(self.batchsize, tf.float32)
        inti_fw = lstm_cell_bw.zero_state(self.batchsize, tf.float32)
        (rnn0, output_states) = tf.nn.bidirectional_dynamic_rnn(lstm_cell_fw,
                                                                lstm_cell_bw,
                                                                ebmeding,
                                                                initial_state_fw=inti_bw,
                                                                initial_state_bw=inti_fw,
                                                                sequence_length=lis_len,
                                                                dtype=tf.float32)
        # output, state = tf.nn.dynamic_rnn(lstm_cell_fw, ebmeding,
        #                                   initial_state=inti_bw ,
        #                                   sequence_length=lis_len,
        #                                   dtype=tf.float32)
        output_fw, output_bw = rnn0
        rnn1 = tf.concat(values=[output_fw, output_bw], axis=2)
        rnn2 = tf.reshape(rnn1, [-1, self.decode_units*2 ])
        W0 = self.weight_init([self.decode_units*2, self.num_classes])
        b0 = self.bias_init([self.num_classes])
        output = tf.matmul(rnn2, W0) + b0
        unary_scores = tf.reshape(output, [self.batchsize, -1, self.num_classes])
        # tv = tf.trainable_variables()  # 得到所有可以训练的参数，即所有trainable=True 的tf.Variable/tf.get_variable
        # regularization_cost = L2_penalty * tf.reduce_sum([tf.nn.l2_loss(v) for v in tv])  # 0.001是lambda超参数
        with tf.name_scope("loss"):
            if crf:
                log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(unary_scores, labelss,sequence_lengths=lis_len) # need to evaluate it for decoding
                # loss = tf.reduce_mean(-log_likelihood)+regularization_cost
                loss = tf.reduce_mean(-log_likelihood)
                decode_tags, best_score = tf.contrib.crf.crf_decode(unary_scores, trans_params, lis_len)
                label_equal = tf.equal(tf.reshape(decode_tags,[-1]), tf.reshape(labelss, [-1]))
                acc = tf.reduce_mean(tf.cast(label_equal, tf.float32))
                result = tf.reshape(decode_tags,[-1])
            else:
                loss = tf.reduce_mean(
                    # tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(labelss, [-1]), logits=output))+regularization_cost
                    tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(labelss, [-1]), logits=output))
                label_equal = tf.equal(tf.cast(tf.argmax(output, 1), tf.int32), tf.reshape(labelss, [-1]))
                acc = tf.reduce_mean(tf.cast(label_equal, tf.float32))
                result = tf.argmax(output, 1)
            tf.summary.scalar("loss", loss)
        optimize = tf.train.RMSPropOptimizer(self.learning_rate).minimize(loss)
        sess = tf.InteractiveSession()
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        merged = tf.summary.merge_all()
        write = tf.summary.FileWriter(r'logs', sess.graph)
        return sess, optimize, loss, acc, saver, merged, write, result

    def train_epoch(self,model_type):
            print(">>>>>>>>>>>>>Fload:\t", m, ">>>>>>>>>>>>>\n")

            print(">>>>>>>>>>>>>Fload:\t", m, ">>>>>>>>>>>>>")
            if model_type == "LSTM_CRF":
                sess, train_step, loss, acc, saver, merged, write, result = model.LSTM_CRF(inputss,
                                                                                       labelss,drop_keep_rate,
                                                                                       lis_len)
                return sess, train_step, loss, acc, saver, merged, write, result

            elif model_type == "LSTM_attention":
                sess, train_step, loss, acc, saver, merged, write, result = model.LSTM_At(inputss,
                                                                                         labelss,drop_keep_rate,
                                                                                         lis_len)
                return sess, train_step, loss, acc, saver, merged, write, result

            elif model_type == "indrnn":
                sess, train_step, loss, acc, saver, merged, write, result = model.indrnn_model(inputss,
                                                                                         labelss,
                                                                                         lis_len)
                return sess, train_step, loss, acc, saver, merged, write, result



if __name__ == "__main__":
    model=Model(20,"../data/train/", "AGA.tab")
    words_, labels_ = model.load_data()
    flod_F = {}
    flod_P = {}
    flod_R = {}
    Favg_flod = defaultdict(list)
    Pavg_flod = defaultdict(list)
    Ravg_flod = defaultdict(list)
    fl = 5#5折交叉
    for m in range(fl):
        g1=tf.Graph()
        g2=tf.Graph()
        train_x = []
        train_y = []
        test_y = []
        test_x = []
        indexs=[]
        nindexs=[]
        words, labels = train_x, train_y
        for mm ,nn in model.split(words_, n=5):
            indexs.append(mm)
            nindexs.append(nn)
        index = indexs[m]
        nindex = nindexs[m]
        for k in index:
            test_x.append(words_[k])
            test_y.append(labels_[k])
        for i in nindex:
            train_x.append(words_[i])
            train_y.append(labels_[i])
        # train_x.extend(test_x)
        # train_y.extend(test_y)
        # print(len(train_x))
        # print(len(train_y))
        datasize = round(len(train_x))
        with g1.as_default():
            inputss = tf.placeholder(tf.int32, [model.batchsize, None], name="inputss")
            labelss = tf.placeholder(tf.int32, [model.batchsize, None], name="labelss")
            lis_len = tf.placeholder(tf.int32, [model.batchsize], name="truelen")
            drop_keep_rate = tf.placeholder(tf.float32, name="dropout_keep")
            sess, train_step, loss, acc, saver, merged, write, result=model.train_epoch(model.model_type)

            for step in range(1000):
                if step % 200 == 0:
                    words, labels = model.shufflelist(train_x, train_y)
                start = (step * model.batchsize) % datasize
                end = min(start + model.batchsize, datasize)
                if len(words[start:end]) == model.batchsize:
                    x_ = words[start:end]
                    y_ = labels[start:end]
                    x,lis_len_x = model.pooling(x_)
                    y ,lis_len_y= model.pooling(y_)
                    sess.run(train_step, feed_dict={inputss: x, labelss: y,lis_len:lis_len_x,drop_keep_rate:0.7})
                    #每训练两百轮评价一次
                    if step % 50== 0:
                        ls, ac, re = sess.run([loss, acc, result], feed_dict={inputss: x, labelss: y,
                                                                            lis_len:lis_len_x,
                                                                            drop_keep_rate:0.7})
                        graph = sess.run(merged, feed_dict={inputss: x, labelss: y,
                                                            lis_len:lis_len_x,
                                                            drop_keep_rate:0.7})
                        write.add_summary(graph, step)
                        print("EPOCH:", step, "\t Loss:", ls, "\t Acc", ac)
                saver.save(sess,"../model/model.ckpt")
    #     with g2.as_default():
    #         inputss = tf.placeholder(tf.int32, [model.batchsize, None], name="inputss")
    #         labelss = tf.placeholder(tf.int32, [model.batchsize, None], name="labelss")
    #         lis_len = tf.placeholder(tf.int32, [model.batchsize], name="truelen")
    #         drop_keep_rate = tf.placeholder(tf.float32, name="dropout_keep")
    #         sess2, train_step, loss, acc, saver, merged, write, result=model.train_epoch(model.model_type)
    #         saver.restore(sess2,"../model/model.ckpt")
    #         FF, PP, RR=model.eval_result(sess2,lis_len, result, inputss, 0, 1, model.batchsize,test_x, test_y)
    #         for key, value in FF.items():
    #             flod_F[key]=value
    #         for key, value in PP.items():
    #             flod_P[key] = value
    #         for key, value in RR.items():
    #             flod_R[key] = value
    # #     for key ,value in flod_F.items():
    # #         Favg_flod[key].append(value)
    # #     for key ,value in flod_P.items():
    # #         Pavg_flod[key].append(value)
    # #     for key ,value in flod_R.items():
    # #         Ravg_flod[key].append(value)
    # # print(">>>>>>>>{} flod>>>>>>>>".format(fl))
    # # # print(avg_flod)
    # # for key,value in Favg_flod.items():
    # #     avg = sum(value)/len(value)
    # #     print("Averge {}:\t\tF:{}\t".format(key,avg))
    # # for key,value in Pavg_flod.items():
    # #     avg = sum(value)/len(value)
    # #     print("Averge {}:\t\tP:{}\t".format(key,avg))
    # # for key,value in Ravg_flod.items():
    # #     avg = sum(value)/len(value)
    # #     print("Averge {}:\t\tR:{}\t".format(key,avg))
    #
