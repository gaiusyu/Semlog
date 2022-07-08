import re

import test9 as SB
#SB.get_Percentage('../SaveFiles&Output/dataset/HDFS/HDFScontent.csv','HDFS',0.2)
#SB.get_Percentage('../SaveFiles&Output/dataset/HDFS/HDFS2k.log','HDFS',0.4)
#SB.get_Percentage('../SaveFiles&Output/dataset/HDFS/HDFS2k.log','HDFS',0.6)



def unseen_word_count(input_path,check_path):
    lines = open(input_path)  # 加载embedding
    line = lines.read()
    lines.close()
    lines1 = open(check_path)  # 加载embedding
    line1 = lines1.read()
    lines1.close()
    generate=re.split(r' +', line)

    raw =re.split(r' +', line1)
    count=0
    raw=list(set(raw))
    print(len(raw))
    for r in raw:
        if not r in generate:
            count+=1

    print(count)
#SB.get_Percentage('../SaveFiles&Output/dataset/Spark/Sparkcontent.csv','Spark_part',0.6)
#SB.get_Percentage('../SaveFiles&Output/dataset/HDFS/HDFScontent.csv','HDFS_part',0.4)
#SB.get_Percentage('../SaveFiles&Output/dataset/HDFS/HDFS2k.log','HDFS',0.6)
#unseen_word_count('../SaveFiles&Output/Cluster/HDFS_part/HDFS_part0.4P.csv','../SaveFiles&Output/dataset/HDFS/HDFScontent.csv')
SB.clusteringwithlen('/Cluster/HDFS_part/HDFS_part0.4P.csv', 'HDFS_part')
#SB.train_parse('HDFS_part','../SaveFiles&Output/Cluster/HDFS_part/array_len.csv')

#SB.get_Percentage('../SaveFiles&Output/dataset/Proxifier/Proxifiercontent.csv','Proxifier',0.6)
#SB.get_Percentage('../SaveFiles&Output/dataset/HDFS/HDFS2k.log','HDFS',0.4)
#SB.get_Percentage('../SaveFiles&Output/dataset/HDFS/HDFS2k.log','HDFS',0.6)
#unseen_word_count('../SaveFiles&Output/Cluster/Proxifier/Proxifier0.4P.csv','../SaveFiles&Output/dataset/Proxifier/Proxifiercontent.csv')
#SB.clusteringwithlen('../SaveFiles&Output/Cluster/Proxifier/Proxifier0.2P.csv','Proxifier')

#SB.get_Percentage('../SaveFiles&Output/dataset/Spark/Sparkcontent.csv','Spark_part',0.05)
#SB.train_parse('Spark_part','../SaveFiles&Output/Cluster/Spark_part/array_len.csv')
#unseen_word_count('../SaveFiles&Output/Cluster/Spark_part/Spark_part0.4P.csv','../SaveFiles&Output/dataset/Spark/Sparkcontent.csv')
#SB.clusteringwithlen('../SaveFiles&Output/Cluster/Spark_part/Spark_part0.4P.csv','Spark_part')
#SB.train_parse('Spark_part','../SaveFiles&Output/Cluster/Spark_part/array_len.csv')

#SB.getcontentwithoutnumber('dataset/HealthyApp/HealthyApp.log', 'dataset/HealthyApp/HealthyAppcontent.csv', r'^.+\|(.+)$', 'num')
#SB.get_Percentage('../SaveFiles&Output/dataset/HealthyApp/HealthyAppcontent.csv','HealthyApp_part',0.4)
#unseen_word_count('../SaveFiles&Output/Cluster/HealthyApp_part/HealthyApp_part0.4P.csv','../SaveFiles&Output/dataset/HealthyApp/HealthyAppcontent.csv')
#SB.clusteringwithlen('../SaveFiles&Output/Cluster/HealthyApp_part/HealthyApp_part0.4P.csv','HealthyApp_part')
#SB.train_parse('HealthyApp_part','../SaveFiles&Output/Cluster/HealthyApp_part/array_len.csv')