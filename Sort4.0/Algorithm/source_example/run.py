#coding:utf-8
#使用docsim方法：doc2bow、similarities判断相似性
from gensim import models,corpora,similarities
import jieba.posseg as pseg
import os
root_path = os.getcwd()
def a_sub_b(a,b):
    ret = []
    for el in a:
        if el not in b:
            ret.append(el)
    return ret
    
#读取文件
raw_documents=[]
walk = os.walk(root_path+"/"+"new_class")
i = 0
for root, dirs, files in walk:
    print(i)
    i = i + 1
    for name in files:
        f = open(os.path.join(root, name), 'r',encoding = 'utf-8')
        raw = str(name)+"\n"
        raw += f.read()
        raw_documents.append(raw)
# stop = [line.strip().decode('utf-8') for line in open('stopword.txt').readlines() ]
#创建语料库
corpora_documents = []
for item_text in raw_documents:
    item_str=[]
    item= (pseg.cut(item_text)) #使用jieba分词
    for i in list(item):
        item_str.append(i.word)
    # item_str=a_sub_b(item_str,list(stop))
    corpora_documents.append(item_str)

# 生成字典和向量语料
dictionary = corpora.Dictionary(corpora_documents) #把所有单词取一个set，并对set中每一个单词分配一个id号的map
corpus = [dictionary.doc2bow(text) for text in corpora_documents]  #把文档doc变成一个稀疏向量，[(0,1),(1,1)]表明id为0,1的词出现了1次，其他未出现。
similarity = similarities.Similarity('-Similarity-index', corpus, num_features=999999999)

test_data_1 = '随州供电公司110kV杨寨变电站10kV2#电容器组于2007年11月投入运行。2013年12月，检修分公司广水运维站在对其进行红外测温时发现，其电抗器A相红外测温异常，表面温度最高达152度，负荷电流为130A。正常相温度为25度，负荷电流为130A，环境温度为10度。经过停电诊断性试验，发现A相电抗器直流电阻偏大，三相相间互差达117%，超出规程规定的2%，经分析认为，该相电抗器线圈已损坏。经现场检查，发现电抗器内侧有烧蚀痕迹，已制定大修计划择期进行处理，目前处于退运状态。'
test_cut = pseg.cut(test_data_1)
test_cut_raw_1=[]
for i in list(test_cut):
    test_cut_raw_1.append(i.word)
test_corpus_1 = dictionary.doc2bow(test_cut_raw_1)
similarity.num_best = 5
print(similarity[test_corpus_1])  # 返回最相似的样本材料,(index_of_document, similarity) tuples
for i in similarity[test_corpus_1]:
    sim=""
    # print('################################')
    # print(i[0])
    for j in corpora_documents[i[0]]:
        sim+=j
    print(sim.split('\n')[0])
