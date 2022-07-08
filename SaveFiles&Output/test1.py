
import test9 as SB
#Scrub.train_certain_cluster('HDFS','21')
#Scrub.parse_certain_cluster('HDFS','21')
#getcontentwithoutnumber('../SaveFiles&Output/dataset/HDFS/HDFS2k.log','../SaveFiles&Output/dataset/HDFS/HDFSwithoutnumber.csv',r'^.+?: (.+)$','nonum')

#SB.getcontentwithoutnumber('../SaveFiles&Output/dataset/Apache/Apache.log','../SaveFiles&Output/dataset/Apache/Apachecontent.csv',r'^.+?\] .+?\] (.+)$','num')
#SB.getcontentwithoutnumber('../SaveFiles&Output/dataset/Proxifier/Proxifier.log','../SaveFiles&Output/dataset/Proxifier/Proxifiercontent.csv',r'^.+?\- (.+)$','num')
#SB.getcontentwithoutnumber('../SaveFiles&Output/dataset/HealthyApp/HealthApp.log','../SaveFiles&Output/dataset/HealthyApp/HealthyAppcontent.csv',r'^.+\|(.+)$','num')
#SB.clusteringwithlen('../SaveFiles&Output/dataset/Proxifier/Proxifierwithoutnumber.csv','../SaveFiles&Output/dataset/Proxifier/Proxifiercontent.csv', 'Proxifier')
#SB.clusteringwithlen('../SaveFiles&Output/dataset/HealthyApp/HealthyAppcontent.csv', 'HealthyApp')
#SB.train_certain_cluster('HDFS','20P')
#SB.parse_certain_cluster('Apache',88)
#SB.train_certain_cluster('Apache',88)
SB.getcontentwithoutnumber('dataset/HDFS/HDFS2k.log', '/dataset/HDFS/HDFScontent.csv', r'^.+?: (.+)$', 'num')
#SB.getcontentwithoutnumber('../SaveFiles&Output/dataset/Android/Android.log','../SaveFiles&Output/dataset/Android/Androidcontent.csv',r'^.+?: (.+)$','num')
#SB.clusteringwithlen('../SaveFiles&Output/dataset/Android/Androidcontent.csv', 'HealthyApp')
#SB.get_20P('../SaveFiles&Output/dataset/HDFS/HDFScontent.csv','HDFS')
#SB.train_parse('Apache','../SaveFiles&Output/Cluster/Apache/array_len.csv')
#SB.parse_certain_cluster('Proxifier',13)
SB.clusteringwithlen('/dataset/HDFS/HDFScontent.csv', 'HDFS')

#SB.train_parse('BGL','../SaveFiles&Output/Cluster/BGL/array_len.csv')
'''
{  Proxifier:
rgex: r'^.+?\- (.+)$'

HealthyApp:
rgex:r'^.+|(.+)$'


HDFS:
rgex:r'^.+?: (.+)$


  Apache:
  rgex: r'^.+?\] .+?\] (.+)$
}

'''