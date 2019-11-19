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
# vis = visdom.Visdom()
data_root = os.path.dirname(__file__)+"/"
print(data_root)
print(os.listdir(data_root))
for file in os.listdir(data_root+"dict/"):
    jieba.load_userdict(data_root+"dict/"+file)

class Option(object):
    lr = 0.005
    epochs = 6000
    batch_size = 32
    early_stopping = 3000
    dropout = 0.5
    max_norm = 3.0
    embedding_dim = 100
    filter_sizes = [3,4,5]
    filter_num = 100
    static = False
    non_static = True
    pretrained_name = 'elec.word'
    pretrained_path = '/home/wzx/link/TextCNN/mydata'
    device = 0
    log_interval = 1
    test_interval = 100
    save_best = False
    class_num=30
    vocabulary_size=5327

class TextCNN(nn.Module):
    def __init__(self, args):
        super(TextCNN, self).__init__()
        self.args = args

        class_num = args.class_num
        chanel_num = 1
        # 卷积核个数
        filter_num = args.filter_num
        # 卷积核大小
        filter_sizes = args.filter_sizes
        vocabulary_size = args.vocabulary_size
        # 嵌入维度
        embedding_dimension = args.embedding_dim
        self.embedding = nn.Embedding(vocabulary_size, embedding_dimension)

        # 加载词向量
        if args.static:
            self.embedding = self.embedding.from_pretrained(args.vectors, freeze=not args.non_static)

        self.convs = nn.ModuleList([nn.Conv2d(1, filter_num, (size, embedding_dimension)) for size in filter_sizes])
        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(len(filter_sizes) * filter_num, class_num)

    def forward(self, x):
        #         print(x.shape)
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

#加载词向量
def load_word_vector(model_name,model_path):
    vectors = Vectors(name = model_name,cache = model_path)
    return vectors
#切词 + 过滤
def word_cut(text):
    regex = re.compile(r'[^\u4e00-\u9fa5s]')
    text = regex.sub('',text)
    return [word for word in jieba.cut(text) if word.strip()]

def predict_err(input_content):
    model_t = TextCNN(opt)
    model_t.load_state_dict(torch.load(data_root+"checkpoints/TextCNN_best",map_location='cpu'))
    model_t.eval()
    with open(data_root+"train_stoi_itos.pickle","rb") as f:
        train_si = pickle.load(f)
    dict_stoi,dict_itos = train_si["stoi"],train_si["itos"]
    pclass = list(pd.read_csv(data_root+"class.csv",encoding = 'GB2312')["errname"])
    input_content = torch.tensor(list(map(lambda x : word2i(x,dict_stoi),word_cut(input_content))))
    tmp=[]
    tmp.append(1)
    input_content = torch.cat((input_content,torch.tensor(tmp)))
    res = model_t(input_content.unsqueeze(0)).squeeze(0)
    label=torch.argmax(res)
    ans = pclass[label-1]
    return ans

def word2i(w,d):
    if(w not in d.keys()):
        return 0
    return d[w]

opt = Option()
# text="国网新疆电力公司哈密供电公司110kV某变电站1号主变压器，型号为SFSZ8-20000/110，生产日期为1996年12月，1998年3月投入运行。2010年5月20日检测人员用超声波局部放电检测仪发现1号主变压器110kV高压侧套管C相下方主变压器本体处存在放电信号。2010年5月23日例行停电检修时发现该变压器35kV绕组1-5档直阻三相不平衡率超标。通过检修孔检查发现中压无载分接开关触头表面存在烧熔、放电情况。现场吊芯处理后，恢复送电缺陷消除。（1）超声波检测2010年5月20日，检测人员进行超声波局部放电检测时，发现该主变压器110kV高压侧套管C相下方主变压器本体处存在放电信号，具体测试点见图1、测试图谱见图2。在图2中，图a显示峰值集中于特征指数为1，同时声脉冲的数量也较多，此时的图像显示为一个尖的凸起；图b显示有50dB的放电量，且放电连续发生，说明局放能量较大；图c中电压对时间的长波形也呈现明显的局放特性；图d显示散点图随时间集中于声脉冲特征指数为1的位置，表明存在明显的局部放电。（2）停电试验对该主变压器例行试验时发现该主变压器中压侧1-5档直阻三相不平衡率超标，但其它试验数据均合格。直阻测试数据见表1通过检修孔检查发现中压无载分接开关触头表面存在明显放电痕迹，分接开关触头弹簧压力不够，造成接触不良，导致触头放电，如图3。随后将该主变压器进行现场吊芯，对分接开关触头弹簧压紧处理后，恢复送电，缺陷消除。"
# predict_err(text)