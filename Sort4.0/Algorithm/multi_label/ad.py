# coding=utf-8
from sys import argv
data_root = "D:/study/EN/Electric-Err1/Algorithm/multi_label/"
stxt = ""
with open(data_root+"t.txt","r",encoding = 'utf-8') as f:
   stxt = f.read()
with open(data_root+"s.txt","w",encoding = 'utf-8') as f:
   f.write(stxt)
