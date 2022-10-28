import Semlog as SB
import re
def generate_partial(input_path,data,percent,regex,delimeter):
    lines = open(input_path+'/'+data+'_2k.log')  # 加载embedding
    line = lines.read()
    lines.close()
    sentences=line.split('\n')
    length=len(sentences)
    num=percent*length
    i=0
    partial_data=''
    for s in sentences:
        i+=1
        if i>num:
            break
        partial_data=partial_data+(s)+'\n'
    f=open(input_path+'/'+data+str(percent)+'.log','w')
    f.write(partial_data)
    log_test,log_train,OOV=OOVcount(regex,data,percent)
    batch=SB.makedata2(log_train,'NotEmbedding',0,delimeter)
    SB.train2(batch,10,data,data,percent,0.01)
    PA, template, template_num, template_set = SB.Parse_with_scrub1(data, log_test, delimeter,percent)
    print('Parsing accuracy='+str(PA))
    print('Out of vocabulary word='+str(OOV))
def OOVcount(regex,dataset,percent):
    parse = SB.format_log(regex, '../logs/'+dataset)
    form = parse.format(dataset+'_2k.log')
    content_str = form['Content'].to_numpy()
    content = '\n'.join(i for i in content_str)
    parse1 = SB.format_log(regex, '../logs/'+dataset)
    form1 = parse1.format(dataset+str(percent)+'.log')
    content1_str = form1['Content'].to_numpy()
    content1 = '\n'.join(i for i in content1_str)
    logs=re.split(r' +', content)
    logs1=re.split(r' +', content1)
    OOV=0
    for word in logs:
        if word not in content1:
            OOV+=1
    print(str(OOV))
    return content, content1,OOV


generate_partial('../logs/Apache','Apache',0.05,'\[<Time>\] \[<Level>\] <Content>',['[,!?=]'])
