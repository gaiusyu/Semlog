import pandas as pd

from Semlog import Semlog as SB
from Semlog import Semlog as SB




print('Hello')
benchmark_settings = {

'Android': {
        'log_file': 'Android/Android_2k.log',
        'log_format': '<Date> <Time>  <Pid>  <Tid> <Level> <Component>: <Content>',
        'delimiter': ['[,?_:]'],

        },

 'Apache': {
        'log_file': 'Apache/Apache_2k.log',
        'log_format': '\[<Time>\] \[<Level>\] <Content>',
        'delimiter': ['[,!?=]']
    },

    'BGL': {
        'log_file': 'BGL/BGL_2k.log',
        'log_format': '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>',
        'delimiter': ['[,!?=]']
    },

    'Hadoop': {
        'log_file': 'Hadoop/Hadoop_2k.log',
        'log_format': '<Date> <Time> <Level> \[<Process>\] <Component>: <Content>',
        'delimiter': ['[,=]']
    },
    'HDFS': {
        'log_file': 'HDFS/HDFS_2k.log',
        'log_format': '<Date> <Time> <Pid> <Level> <Component>: <Content>',
        'delimiter': ['[,!?=]']
    },
    'HealthApp': {
        'log_file': 'HealthApp/HealthApp_2k.log',
        'log_format': '<Time>\|<Component>\|<Pid>\|<Content>',
        'delimiter': ['[,!?=:]']
    },
    'HPC': {
        'log_file': 'HPC/HPC_2k.log',
        'log_format': '<LogId> <Node> <Component> <State> <Time> <Flag> <Content>',
        'delimiter': ['[,=]']
    },

    'Linux': {
        'log_file': 'Linux/Linux_2k.log',
        'log_format': '<Month> <Date> <Time> <Level> <Component>(\[<PID>\])?: <Content>',
        'delimiter': ['[,!?=]']
    },

    'Mac': {
        'log_file': 'Mac/Mac_2k.log',
        'log_format': '<Month>  <Date> <Time> <User> <Component>\[<PID>\]( \(<Address>\))?: <Content>',
        'delimiter': ['[,\[\]]']
    },

    'OpenSSH': {
        'log_file': 'OpenSSH/OpenSSH_2k.log',
        'log_format': '<Date> <Day> <Time> <Component> sshd\[<Pid>\]: <Content>',
        'delimiter': ['[,_]']
    },

    'OpenStack': {
        'log_file': 'OpenStack/OpenStack_2k.log',
        'log_format': '<Logrecord> <Date> <Time> <Pid> <Level> <Component> \[<ADDR>\] <Content>',
        'delimiter': ['[,.]']
    },


'Proxifier': {
        'log_file': 'Proxifier/Proxifier_2k.log',
        'log_format': '\[<Time>\] <Program> - <Content>',
        'delimiter': ['[,!?=]']
    },
    'Spark': {
        'log_file': 'Spark/Spark_2k.log',
        'log_format': '<Date> <Time> <Level> <Component>: <Content>',
        'delimiter': ['[,!?=]']
        },

    'Thunderbird': {
        'log_file': 'Thunderbird/Thunderbird_2k.log',
        'log_format': '<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component>(\[<PID>\])?: <Content>',
        'delimiter': ['[,!?=]']
        },

    'Windows': {
        'log_file': 'Windows/Windows_2k.log',
        'log_format': '<Date> <Time>, <Level>                  <Component>    <Content>',
        'delimiter': ['[,!?=]']
        },
    'Zookeeper': {
        'log_file': 'Zookeeper/Zookeeper_2k.log',
        'log_format': '<Date> <Time> - <Level>  \[<Node>:<Component>@<Id>\] - <Content>',
        'delimiter': ['[.,!?]']
    },






}


bechmark_result = []

PC=[]
for dataset, setting in benchmark_settings.items():
    parse = SB.format_log(
        log_format=setting['log_format'],
        indir='../logs/'+dataset)
    form = parse.format(dataset+'_2k.log')

    content = form['Content']
    # logID = form['LineId']
    # Date = form['Date']
    # Time = form['Time']
    arr = content.to_numpy()
    PA,template,template_num,template_set=SB.Parse_with_scrub1(dataset, arr, setting['delimiter'])
    form['template']=template
    form['template_num']=template_num
    form.to_csv('../SaveFiles&Output/Parseresult/' + dataset + '/' + dataset + 'result.csv', index=False)
    with open('../SaveFiles&Output/Parseresult/' + dataset + '/' + dataset + 'templates.csv', 'w') as f:
        template_num = 0
        for k in template_set:
            f.write(' '.join(k))
            f.write('\n')
        f.close()
    lenth=len(form)

    PC.append(setting['log_file']+'     '+str(PA))
print('####### % Parsing Accuracy % ########'+'\n')
for pa_ in PC:
    print(str(pa_))
print('ok')

'''
'''