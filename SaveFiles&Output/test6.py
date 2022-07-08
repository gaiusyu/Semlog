import re

import test12 as SB

#SB.getcontentwithoutnumber('dataset/Thunderbird/Thunderbird.log', 'dataset/Thunderbird/Thunderbirdcontent.csv', r'^.+?: (.+)$', 'num')
#SB.clusteringwithlen('dataset/Thunderbird/Thunderbirdcontent.csv', 'Thunderbird')
#SB.train_parse('Thunderbird','../SaveFiles&Output/Cluster/Thunderbird/array_len.csv')
#SB.getcontentwithoutnumber('dataset/Mac/Mac.log', 'dataset/Mac/Maccontent.csv', r'^.+?\]: (.+)$', 'num')
#SB.clusteringwithlen('dataset/Mac/Maccontent.csv', 'Mac')
#SB.train_parse('Mac','../SaveFiles&Output/Cluster/Mac/array_len.csv')
#SB.getcontentwithoutnumber('dataset/OpenStack/OpenStack.log', 'dataset/OpenStack/OpenStackcontent.csv', r'^.+?\] (.+)$', 'num')
#SB.clusteringwithlen('dataset/OpenStack/OpenStackcontent.csv', 'OpenStack')
#SB.train_parse('OpenStack','../SaveFiles&Output/Cluster/OpenStack/array_len.csv')
#SB.parse_certain_cluster('Proxifier',7)
#SB.getcontentwithoutnumber('dataset/HDFS/HDFS.log', 'dataset/HDFS/HDFScontent.csv', r'^.+?: (.+)$', 'num')
#SB.clusteringwithlen('dataset/HDFS/HDFScontent.csv', 'HDFS')
SB.parse_certain_cluster('Apache',999)
'''

lines = open('../SaveFiles&Output/Cluster/Apache/Apache8.csv', encoding='UTF-8')
line = lines.read()
lines.close()
count=0
sentences = line.split('\n')
for sentence in sentences:
        content = re.split(' +', sentence)
        if content[0] != 'terminating':
            count+=1
            print(count)

with open('Cluster/Apache/Apache999.csv', 'w', encoding='UTF-8') as f:
    for sentence in sentences:
        content = re.split(' +', sentence)
        if len(content)<=1:
            continue
        c = content[1]
        if content[1] == '[client':
            count += 1
            f.write(sentence+'\n')



'''