from Semlog import Semlog as SB

print('Hello')
benchmark_settings = {

    'Mac': {
        'log_file': 'Mac',
        'get_content': [r'^.+?\]: (.+)$'],
        'delimiter': ['[,\[\]]']
    },
    'HPC': {
        'log_file': 'HPC',
        'get_content': [r'^.+? .+? .+? .+? .+? .+? (.+)$'],
        'delimiter': ['[,!?\\-_=]']
    },

    'HDFS': {
        'log_file': 'HDFS',
        'get_content': [r'^.+?: (.+)$'],
        'delimiter': ['[,!?=]']
    },

    'Hadoop': {
        'log_file': 'Hadoop',
        'get_content': [r'^.+?\].+?: (.+)$'],
        'delimiter': ['[,!?=]']
    },

    'BGL': {
        'log_file': 'BGL',
        'get_content': [r'^.+? .+? .+? .+? .+? .+? .+? .+? .+? (.+)$'],
        'delimiter': ['[.,!?=:]']
    },

    'Spark': {
        'log_file': 'Spark',
        'get_content': [r'^.+?: (.+)$'],
        'delimiter': ['[,!?=:]']
    },


    'Zookeeper': {
        'log_file': 'Zookeeper',
        'get_content': [r'^.+?\] - (.+)$'],
        'delimiter': ['[,!?\\-_=:]']
    },


    'Linux': {
        'log_file': 'Linux',
        'get_content': [r'^.+?: (.+)$'],
        'delimiter': ['[,!?=:]']
    },

    'Android': {
        'log_file': 'Android',
        'get_content': [r'^.+?: (.+)$'],
        'delimiter': ['[,!?\\-_=:]']
    },

    'HealthApp': {
        'log_file': 'HealthApp',
        'get_content': [r'^.+\|(.+)$'],
        'delimiter': ['[,!?=:]']
    },

    'Apache': {
        'log_file': 'Apache',
        'get_content': [r'^.+?\] .+?\] (.+)$'],
        'delimiter': ['[,!?\\-_=]']
    },

    'Proxifier': {
        'log_file': 'Proxifier',
        'get_content': [r'^.+?\- (.+)$'],
        'delimiter': ['[,!?=]']
    },

    'OpenSSH': {
        'log_file': 'OpenSSH',
        'get_content': [r'^.+?\]: (.+)$'],
        'delimiter': ['[,!?=\\-:]']
    },

    'OpenStack': {
        'log_file': 'OpenStack',
        'get_content': [r'^.+?\] (.+)$'],
        'delimiter': ['[,\[\]]']
    },

    'Thunderbird': {
        'log_file': 'Thunderbird',
        'get_content': [r'^.+?: (.+)$'],
        'delimiter': ['[,\[\]]']
    },

    'Windows': {
        'log_file': 'Windows',
        'get_content': [r'^.+[a-z]    (.+)$'],
        'delimiter': ['[,]']
    },


}


bechmark_result = []

PC=[]
for dataset, setting in benchmark_settings.items():
    PA=SB.Parse_with_scrub(setting['log_file'], setting['get_content'], setting['delimiter'])
    PC.append(setting['log_file']+'     '+str(PA))
print('####### % Parsing Accuracy % ########'+'\n')
for pa_ in PC:
    print(str(pa_))
print('ok')

'''
'''