import re

def clusteringwithlen(content_path, output):
    i = 0
    n = 0
    b2 = open(content_path, encoding='UTF-8')  # 加载embedding
    line2 = b2.read()
    b2.close()
    sentences = line2.split('\n')
    array_len = []

    for sentence in sentences:
        sentence1 = []
        sentence2 = sentence
        sentence1.append(re.split(' +', sentence))
        sentence = sentence1[0]

        lent = len(sentence)
        if lent <= 1:
            continue
        if str(lent) not in array_len:
            array_len.append(str(len(sentence)))
            with open('../SaveFiles&Output/Cluster/' + str(output) + '/array_len.csv', 'a+', encoding='UTF-8') as f:
                f.write(str(len(sentence)) + '\n')
                f.close()
            i += 1
        f = open('../SaveFiles&Output/Cluster/' + str(output) + '/' + str(output) + str(len(sentence)) + '.csv',
                 'a+', encoding='UTF-8')  # 使用‘a'来提醒python用附加模式的方式打开
        f.write(sentence2 + '\n')
        f.close()
        n += 1
clusteringwithlen('/dataset/HDFS/HDFScontent.csv', 'HDFS')