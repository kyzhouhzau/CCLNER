#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:zhoukaiyin
@Time:2018/5/12
"""
import pickle
import gensim
import numpy as np
import sys
def word2id(base_path,path):
    vocabs=[]
    labels=[]
    rf=open(base_path+path)
    filew2id = open(base_path+'w2id.pkl','wb')
    fileid2w = open(base_path+'id2w.pkl','wb')
    filelabel2id = open(base_path+'label2id.pkl','wb')
    fileid2label = open(base_path+'id2label.pkl','wb')
    for line in rf:
        if len(line)>1:
            line_pure = line.strip()
            line_list = line_pure.split("\t")
            vocabs.append(line_list[0])
            labels.append(line_list[-1])
    vocab = list(set(vocabs))
    label = list(set(labels))
    w2id = {w:i for i,w in enumerate(vocab,1)}
    w2id['<start>']=len(vocab)+1
    w2id['<end>']=len(vocab)+2
    id2w= {i:w for i,w in enumerate(vocab,1)}
    id2w[len(vocab)+1]='<start>'
    id2w[len(vocab)+2]='<end>'
    label2id = {l: i for i, l in enumerate(label, 1)}
    id2label = {i: l for i, l in enumerate(label, 1)}
    pickle.dump(w2id,filew2id)
    pickle.dump(id2w,fileid2w)
    pickle.dump(label2id,filelabel2id)
    pickle.dump(id2label,fileid2label)
    rf.close()
    fileid2w.close()
    filew2id.close()
    fileid2label.close()
    filelabel2id.close()

def w_embed(base_path):
    li = []
    f = open(base_path+"id2w.pkl",'rb')
    wf = open(base_path+"w2embed.pkl",'wb')
    result = pickle.load(f,encoding='bytes')
    model = gensim.models.KeyedVectors.load_word2vec_format("../data/pubvector.txt",binary=False)
    dim=model.vector_size
    add = np.random.randn(dim,)
    li.append((add-np.mean(add))/len(add))
    for i in range(1,len(result)+1):
        word = result[i].lower()
        try:
            embed = model[word]
        except KeyError:
            embed=np.zeros((dim,))
            word_list=word.split('-')
            try:
                for i, w in enumerate(word_list):
                    embed += model[word_list[i]]
            except Exception:
                print(word + ":Cant find their vector!")
                embed = np.random.randn(dim, )
                embed = (embed-np.mean(embed))/len(embed)
        li.append(embed)
    arrayresult = np.array(li)
    print("__",arrayresult.shape)
    pickle.dump(arrayresult,wf)
    f.close()
    wf.close()
def label2embed(base_path):
    kk = []
    f = open(base_path+"id2label.pkl", 'rb')
    wf = open(base_path+"label2embed.pkl", 'wb')
    result = pickle.load(f, encoding='bytes')
    print(result)
    b = [0 for _ in range(1, len(result) + 2)]

    for i in range(1,len(result)+2):
        a = [0 for _ in range(1, len(result) + 2)]
        a[i-1]=1
        kk.append(a)
    kk[0]=b
    print(len(kk))
    pickle.dump( kk, wf)

def test_pkl(path):
    rf=open(path,'rb')
    result = pickle.load(rf,encoding='bytes')
    print(len(result))
    # print(np.array(result))
    rf.close()

if __name__ == "__main__":
    print(sys.argv)
    if len(sys.argv[:])<2:
        print("Using:python data_preprocessing.py [train/test]")
    ty = sys.argv[1]
    if ty=="train":
        base_path = '../data/train/'
        path = "AGA.tab"
        #path_test = base_path+"/id2w.pkl"
        word2id(base_path,path)
        w_embed(base_path)
        label2embed(base_path)
        #test_pkl(path_test)
    elif ty=="test":
        base_path = 'predict/'
        path = "result.txt"
        word2id(base_path,path)
        w_embed(base_path)
        label2embed(base_path)
