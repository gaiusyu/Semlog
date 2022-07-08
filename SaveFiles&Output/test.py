import re

import numpy as np

import test9 as SB
import networkx as nx
G=nx.DiGraph()#创建空的简单有向图
Parsereult={}
contrbution,sentences=SB.parse_graph('Apache',5)
i=0
list1=[]
nodes=[]
lenset={}
cand_template=[]


def get_prelow():
    i = 0
    list1 = []
    list2 = []
    while i < len(contrbution):
        cand_template=[]
        b = max(contrbution[i])
        c = np.argsort(contrbution[i])
        c =reversed(c)
        sentence = re.split(' +', sentences[i])
        if not len(sentence) in lenset:
            lenset.setdefault(lenset[len(sentence)],[]).append(str(sentence))

        if len(lenset[len(sentence)])==1 and sentence[b] in lenset[len(sentence)]:                  #如果词典中该长度的日志语句只有一类，那么直接进行匹配判断
            sentence.append('E'+str(len(sentence))+'first')
        else:
            lenset[len(sentence)].append(sentence)                                                   #不符合同一类日志模板

    print('I dont know how to keep going')
    print('=========== show the result ==========')
    '''
    for item in list1:
        print("%s has occured for %d" % (str(item), list2.count(item)))
    '''
get_prelow()
print('ok')
'''

    for item in list1:
        print("%s has occured for %d" % (str(item), list2.count(item)))
    print('ok')
    '''
#SB.get_eval_metric('../SaveFiles&Output/Parseresult/Proxifier/Proxifier88.csv','../SaveFiles&Output/Parseresult/Proxifier/template.csv')