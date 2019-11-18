# coding=utf-8
from sys import argv
import torch
import torch.nn as nn
import torch.nn.functional as F
#data
import re
from torchtext import data
import jieba
import logging
from torchtext.vocab import Vectors
import sys
import pandas as pd
import os
import time
import torchnet as tnt
import visdom
import numpy as np
import pickle
# data_root = "D:/study/EN/Electric-Err1/Sort4.0/Algorithm/multi_label/"
data_root = os.path.dirname(__file__)+"/"
print("awsl",data_root)
for file in os.listdir(data_root+"entity_dict/"):
    jieba.load_userdict(data_root+"entity_dict/"+file)

class Option(object):
    lr = 0.001
    epochs = 2000 ##
    batch_size = 4
    early_stopping = 1000
    dropout = 0.5
    max_norm = 3.0
    embedding_dim = 100
    filter_sizes = [3,4,5]
    filter_num = 100
    static = False
    non_static = True
    pretrained_name = 'elec.word'
    pretrained_path = '/home/qinye/link/TextCNN/multi_data'
    device = 1
    log_interval = 1
    test_interval = 100
    save_best = False
    class_num = 25
    train_num = 20
    val_num = 22
    percent = 0.9

    vocabulary_size = 1671 # 可能需要修改
class TextCNN(nn.Module):
    def __init__(self, args):
        super(TextCNN, self).__init__()
        self.args = args

        class_num = args.class_num
        chanel_num = 1
        #卷积核个数
        filter_num = args.filter_num
        #卷积核大小
        filter_sizes = args.filter_sizes
        vocabulary_size = args.vocabulary_size
        #嵌入维度
        embedding_dimension = args.embedding_dim
        self.embedding = nn.Embedding(vocabulary_size, embedding_dimension)

        #加载词向量
        if args.static:
            self.embedding = self.embedding.from_pretrained(args.vectors, freeze=not args.non_static)

        self.convs = nn.ModuleList([nn.Conv2d(1, filter_num, (size, embedding_dimension)) for size in filter_sizes])
        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(len(filter_sizes)*filter_num, class_num)

    def forward(self, x):
        #         print(x.shape)
        x = x.long()
        x = self.embedding(x)
        #         print(x.shape)
        x = x.unsqueeze(1)
        #         print(x.shape)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        #         print(x[0].shape,x[1].shape,x[2].shape)
        x = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x]
        #         print(x[0].shape)#54, 100
        x = torch.cat(x, 1)
        #         print(x.shape)#
        x = self.dropout(x)
        #         print(x.shape)
        logits = self.fc(x)
        logits = torch.sigmoid(logits)
        return logits

#切词 + 过滤
def word_cut(text):
    regex = re.compile(r'[^\u4e00-\u9fa5s]')
    text = regex.sub('',text)
    return [word for word in jieba.cut(text) if word.strip()]
def test(input_content):
    model_t = TextCNN(opt)
    model_t.load_state_dict(torch.load(data_root+"checkpoints/TextCNN_best",map_location='cpu'))
    model_t.eval()
    with open(data_root+"train_stoi_itos.pickle","rb") as f:
        train_si = pickle.load(f)
    dict_stoi,dict_itos = train_si["stoi"],train_si["itos"]
    pclass = list(pd.read_csv(data_root+"class.csv",encoding = 'GB2312')["advice"])
    input_content = torch.tensor(list(map(lambda x : word2i(x,dict_stoi),word_cut(input_content))))
    # print("awsl",input_content)
    res = model_t(input_content.unsqueeze(0)).squeeze(0)
    res = (res.float()/res.float().sum()).sort(descending = True)

    res1,res2 = res[0],res[1]
    ss = 0
    for i in range(len(res1)):
        ss += res1[i]
        if(ss > opt.percent):
            break
    j = min(i+1,len(res1))
    res2 = res2[:j]
    ans = list(map(lambda x : pclass[x],res2))

    res = ""
    for i,j in enumerate(ans):
       idx = i+1
       res += str(idx) + "." + j + "\n"
    return res
def word2i(w,d):
    if(w not in d.keys()):
        return 0
    return d[w]
def i2word(i,l):
    return l[i]

opt = Option()


