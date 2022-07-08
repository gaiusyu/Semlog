import os
import re
import numpy as np
import test9 as SB
def parsing_to_group(contrbution,sentences,type,number):
    i = 0
    lenset = {}
    result = []
    len_t=int(number)
    while i < len(contrbution):
        sentence = []
        c = np.argsort(contrbution[i])
        c = list(reversed(c))
        event = []
        sentence.append(re.split(' +', sentences[i]))
        sentence = sentence[0]
        i += 1
        output, lenset = get_prelow(sentence, c, lenset,len_t)   ##输出解析结果
        result.append(output)
        print('========%%  you are processing %%===' + str(i) + ' LOG')
    li = lenset[len_t]
    n = len(li)
    for i in range(n):
        li = lenset[len_t]
        li = li[i]
        first = li[len_t]
        with open('../SaveFiles&Output/Parseresult/'+str(type)+'/CCCCCCLEN'+str(len_t)+str(type) +'result' + str(i) + '.csv', 'a+') as f:
            a = 0
            for res in result:
                a += 1
                if res[len_t] == first:
                    f.write(str(res) + '\n')
                    print('something wrong happened' + str(a))

        f.close()
    print('ok')
def get_prelow(sentence,c,lenset,len_t):      # 输入待解析的日志语句 sentec     该sentence的语义得分数组c    需要在哪些模板中去判断lenset
        event=[]
        if not len(sentence) in lenset:
            lenset.setdefault(len(sentence),[]).append(sentence)
            sentence.append('Lenth' + str(len(sentence)) + 'Event  ' + str(sentence))
            return sentence,lenset
        count=0
        for i in lenset[len(sentence)]:  #  i表示该长度的日志语句有哪些。
           if sentence[c[0]+1] == i[c[0]+1]:                  #如果词典中该长度的日志语句只有一类，那么直接进行匹配判断
              count+=1
              event.append(i)
        output,lenset=findtemplate(count,event,sentence,c,lenset,len_t)
        if count !=1:
            print('%%%%%%%%%% %%%%%%%%%%%%%%%')
        return output,lenset
def findtemplate(count,event,sentence,c,lenset,len_t):
    output=[]
    event1=[]
    if count == 1:
        event=event[0]
        sentence.append(str(event[len(event)-1]))
        output=sentence
    if count == 0:
        lenset[len(sentence)].append(sentence)# 不符合同一类日志模板
        sentence.append('Lenth' + str(len(sentence)) + 'Event  ' + str(sentence))
        output=sentence
    if count > 1:
        c = c[1:]
        count=0

        for i in event:  # i表示该长度的日志语句有哪些。
            if sentence[c[0]+1] in i:  # 如果词典中该长度的日志语句只有一类，那么直接进行匹配判断
                count += 1
                event1.append(i)
        output,lenset=findtemplate(count,event1,sentence,c,lenset,len_t)
    return output,lenset


def parse_everylen(type):
    lines = open('../SaveFiles&Output/Cluster/'+str(type)+'/array_len.csv', encoding='UTF-8')
    line = lines.read()
    lines.close()
    for number in line.split('\n'):
        if os.path.exists(str('../SaveFiles&Output/Cluster/') + type + '/' + str(type) + number + '.csv') == True:
            print(number)
            contrbution, sentences = SB.parse_graph(type, number)
            parsing_to_group(contrbution, sentences,type,number)
            # parse('../SaveFiles&Output/modelsave/model' + str(type) + number, str('../SaveFiles&Output/Cluster/')+ type + '/'  + str(type) + number + '.csv',type, number)
        else:
            continue
def template_generate(type):
    counts = open('../SaveFiles&Output/Cluster/' + str(type) + '/array_len.csv', encoding='UTF-8')
    count = counts.read()
    counts.close()
    for number in count.split('\n'):
        get_template(type,int(number))


def get_template(type,lenth):
    count=0
    while True:
        if os.path.exists('../SaveFiles&Output/Parseresult/' + type + '/CCCCCCLEN' + str(lenth) + str(type) + 'result' + str(count) + '.csv') == True:
            lines = open('../SaveFiles&Output/Parseresult/' + type + '/CCCCCCLEN' + str(lenth) + str(type) +'result' + str(count) + '.csv',
                         encoding='UTF-8')
            line = lines.read()
            lines.close()
            sentences = line.split('\n')
            template=sentence_to_template(sentences,lenth)

            with open('../SaveFiles&Output/Parseresult/' + type + '/candidateTemplate.csv', 'a+', encoding='UTF-8') as f:
                for t in template:
                    f.write(str(t) + 'LENTH'+str(lenth)+'NUM'+ str(count) + '\n')
                    print(t)
                f.close()
            count += 1
        else:
            break

def sentence_to_template(sentences,lenth):
    result=[]
    template = re.split(' +', sentences[0])
    template = template[1:lenth]
    dict = {}
    for i in range(lenth - 1):
        compare = []
        compare.append(template[i])
        num = 0
        index = 0
        for sentence in sentences:
            sentence = re.split(' +', sentence)
            if len(sentence) <= 1:
                continue
            if num >= 5:
                break
            if not sentence[i + 1] in compare:
                compare.append(sentence[i + 1])
                template[i] = ' <*>'
                num += 1
            dict.setdefault(sentence[i + 1], []).append(index)
            index += 1
        if num >= 5 or num == 0:
            continue
        else:
            for com in compare:
                list = dict[com]
                s = []
                for l in list:
                    s.append(sentences[l])
                template=sentence_to_template(s,lenth)
                if len(template)>=1:
                    for te in template:
                        result.append(te)
                else:
                    result.append(template)
            return result
    result.append(template)
    return result


#template_generate('Apache')
parse_everylen('Android')


'''

    for item in list1:
        print("%s has occured for %d" % (str(item), list2.count(item)))
    print('ok')
    '''
#SB.get_eval_metric('../SaveFiles&Output/Parseresult/Proxifier/Proxifier88.csv','../SaveFiles&Output/Parseresult/Proxifier/template.csv')