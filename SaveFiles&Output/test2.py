import test7
import test9 as SB

#Scrub.train_certain_cluster('HDFS','21')
#Scrub.parse_certain_cluster('HDFS','21')
#getcontentwithoutnumber('../SaveFiles&Output/dataset/HDFS/HDFS2k.log','../SaveFiles&Output/dataset/HDFS/HDFSwithoutnumber.csv',r'^.+?: (.+)$','nonum')

#SB.getcontentwithoutnumber('../SaveFiles&Output/dataset/Zookeeper/Zookeeper.log','../SaveFiles&Output/dataset/Zookeeper/Zookeeperwithoutnumber.csv',r'^.+?\-.+\- (.+)$','nonum')
#SB.getcontentwithoutnumber('../SaveFiles&Output/dataset/Zookeeper/Zookeeper.log','../SaveFiles&Output/dataset/Zookeeper/Zookeepercontent.csv',r'^.+?\-.+\- (.+)$','num')
#SB.parse_certain_cluster('Proxifier',88)
#SB.parse_certain_cluster('Apache',8)
#SB.train_parse('HealthyApp','../SaveFiles&Output/Cluster/HealthyApp/array_len.csv')
#SB.clusteringwithlen('../SaveFiles&Output/Cluster/Apache/Apache88.csv','Apache')
#SB.getcontentwithoutnumber('../SaveFiles&Output/dataset/BGL/BGL.log','../SaveFiles&Output/dataset/BGL/BGLcontent.csv',r'^.+? .+? .+? .+? .+? .+? .+? .+? .+? (.+)$','num')
#SB.clusteringwithlen('../SaveFiles&Output/dataset/BGL/BGLcontent.csv','BGL')
#SB.clusteringwithlen('../SaveFiles&Output/dataset/Proxifier/Proxifiercontent.csv','Proxifier')
#SB.train_parse('Proxifier','../SaveFiles&Output/Cluster/Proxifier/array_len.csv')
#SB.train_certain_cluster('Proxifier',9)
#SB.parse_certain_cluster('Spark',99)
#SB.clusteringwithlen('../SaveFiles&Output/dataset/Android/Androidcontent.csv','Android')
#SB.getcontentwithoutnumber('dataset/Linux/Linux.log', '../SaveFiles&Output/dataset/Linux/Linuxcontent.csv', r'^.+?: (.+)$', 'num')
#SB.clusteringwithlen('../SaveFiles&Output/dataset/Linux/Linuxcontent.csv','Linux')
#SB.getcontentwithoutnumber('dataset/OpenStack/OpenStack.log', '../SaveFiles&Output/dataset/OpenStack/openstackcontent.csv', r'^.+?\] (.+)$', 'num')
#SB.clusteringwithlen('../SaveFiles&Output/dataset/OpenStack/openstackcontent.csv','OpenStack')
#SB.getcontentwithoutnumber('dataset/Mac/Mac.log', '../SaveFiles&Output/dataset/Mac/Maccontent.csv', r'^.+?\]: (.+)$', 'num')
#SB.clusteringwithlen('../SaveFiles&Output/dataset/Mac/Maccontent.csv','Mac')
#SB.getcontentwithoutnumber('dataset/HDFS/HDFS2k.log', '../SaveFiles&Output/dataset/HDFS/HDFScontent.csv', r'^.+?: (.+)$', 'num')
#SB.clusteringwithlen('../SaveFiles&Output/dataset/HDFS/HDFScontent.csv','HDFS')
#SB.getcontentwithoutnumber('../SaveFiles&Output/dataset/Spark/Spark.log','../SaveFiles&Output/dataset/Spark/Sparkcontent.csv',r'^.+?: (.+)$','num')
#SB.clusteringwithlen('../SaveFiles&Output/dataset/Spark/Sparkcontent.csv','Spark')
#SB.train_parse('OpenStack','../SaveFiles&Output/Cluster/OpenStack/array_len.csv')
#SB.getcontentwithoutnumber('dataset/OpenStack/OpenStack.log', 'dataset/OpenStack/OpenStackcontent.csv', r'^.+?\] (.+)$', 'num')
#SB.clusteringwithlen('dataset/OpenStack/OpenStackcontent.csv', 'OpenStack')
#SB.getcontentwithoutnumber('dataset/HDFS/HDFS2k.log', 'dataset/HDFS/HDFScontent.csv', r'^.+?: (.+)$', 'num')
#SB.clusteringwithlen('dataset/Apache/Apachecontent.csv', 'Apache')
SB.train_parse('HDFS','../SaveFiles&Output/Cluster/HDFS/array_len.csv')
'''
{  Proxifier:
rgex: r'^.+?\- (.+)$'

}


{   ZooKeeper
regex: 

BGL:r'^.+? +? +? +? +? +? +? +? +? (.+)$'

Linux: r'^.+?: (.+)$'
}
'''