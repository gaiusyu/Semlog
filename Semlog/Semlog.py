import datetime

from pytorch_pretrained_bert import BertTokenizer
import math
import torch
import random
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import copy

tokenizer = BertTokenizer.from_pretrained('../Bert/bert-base-uncased-vocab.txt')
vocab_size=len(tokenizer.vocab)
maxlen = 512
batch_size =512
max_pred = 5  # 最大被maksed然后预测的个数 max tokens of prediction
n_layers = 1  # encoder的层数
n_heads = 1  # 多头注意力机制头数
d_model = 64  # 中间层维度
d_ff = 64 * 4  # 全连接层的维度 4*d_model, FeedForward dimension
d_k = d_v = 32  # QKV的维度 dimension of K(=Q), V
n_segments = 2  # 一个Batch里面有几个日志语句


torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def file_batch(input_path,delimeter):   #delimeter让空格转变成[PAD]
    lines = open(input_path)
    line = lines.read()
    lines.close()
    sentences = []  #################存储删去源文件索引的日志语句
    sentences2 = line.lower().split('\n')  # filter '.', ',', '?', '!'  re.sub 正则表达式处理数据
    for sentence in sentences2:
        Index = sentence.find(' ')  # 按数据集格式找到该日志的content
        content = sentence[Index + 1:]
        sentences.append(content)
    token_list = list()
    for s in sentences:
        if delimeter == '[PAD]':
            s =',' + re.sub(' ',",",s)
        tokenized_text = tokenizer.tokenize(s)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens])
        k = np.squeeze(tokens_tensor.numpy()).tolist()
        token_list.append(k)
    batch = []
    count_index = 0
    while count_index < len(sentences):

        #########tokens_a_index, tokens_b_index = randrange(len(sentences)), randrange(
        ########   len(sentences))  # sample random index in sentences
        tokens_value = token_list[count_index]
        count_index += 1
        ############tokens_a, tokens_b = token_list[tokens_a_index], token_list[tokens_b_index]
        if isinstance(tokens_value,int):
            tokens_value=[tokens_value]
        input_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("[CLS]")) + tokens_value
        ####nput_ids = [word2idx['[CLS]']] + tokens_a + [word2idx['[SEP]']] + tokens_b + [word2idx['[SEP]']]
        #####　segment_ids = [0] * (1 + len(tokens_a) + 1) + [1] * (len(tokens_b) + 1)

        # MASK LM
        #############################  n_pred = min(max_pred,
        #########################################              max(1, int(len(input_ids) * 0.15)))  # 选择一句话中有多少个token要被mask 15 % of tokens in one sentence
        cand_maked_pos = [i for i, token in enumerate(input_ids)
                          if token != tokenizer.convert_tokens_to_ids(
                tokenizer.tokenize("[CLS]"))]  # 排除分隔的CLS和SEP candidate masked position
        for pos in cand_maked_pos[
                   :len(
                       cand_maked_pos)]:  #########################################for pos in cand_maked_pos[:n_pred]:  # 随机打乱后取前n_pred个token做mask
            masked_tokens, masked_pos = [], []
            masked_pos.append(pos)
            masked_tokens.append(input_ids[pos])
            n_pad = maxlen - len(input_ids)
            input_ids.extend([0] * n_pad)  # 填充操作，每个句子的长度是30 余下部分补000000
            batch.append([input_ids, masked_tokens, masked_pos])
            break
    return batch

def parse(model_path,input_path,output_class,output_token):   # 输入对应的模型路径，需要解析的文件路径，输出到Parseresult文件夹下，output_class为处理文件的类别，Output_token为属于类别的长度
    batch=file_batch(input_path,'No')
    model = vary_bert("parse",vocab_size).to(device)  # 实例化模型
    model.load_state_dict(torch.load(model_path))
    i = 0
    weight = 0.2
    template = []
    batch2 = file_batch(input_path, '[PAD]')
    lines = open(input_path, encoding='UTF-8')
    line = lines.read()
    lines.close()
    sentences = line.lower().split('\n')
    while i < len(batch):
        semantic_contrbution = []
        variable_pos = []
        sc=[]
        n = 1
        l = 0
        input_ids, masked_tokens, masked_pos = batch[i]

        print('================================'+str(i))
        input_ids[masked_pos[0]] = masked_tokens[0]
        attention_score = model(torch.LongTensor([input_ids]),
                                torch.LongTensor([masked_pos]))
        s=input_ids.index(0)
        while n < input_ids.index(0):
            sum1 = 0
            g = 1
            while g < s:
                sum1 += attention_score[g][n]
                g += 1
            semantic_contrbution.append(sum1)  #去掉了[CLS]
            n += 1
        input_ids2, masked_tokens2, masked_pos2 = batch2[i]
        beg=0
        t=0
        b=0
        start=input_ids2.index(1010, 0)
        if not 1010 in input_ids2[start+1:]:
            i+=1
            continue
        while 1010 in input_ids2[beg+1:]:

            beg = input_ids2.index(1010, start+1)
            if beg-start-1 != 0:
               sc.append(sum(semantic_contrbution[start-t-1:beg-2-t])/(beg-start-1))
               b+=1
            t+=1
            start = beg

        if len(semantic_contrbution)-beg+t+1 !=0:
            sc.append(sum(semantic_contrbution[beg-t-1:len(semantic_contrbution)]) / (len(semantic_contrbution)-beg+t+1))
            b=b+1
        sc = nn.Softmax(dim=0)(torch.tensor(sc))
        sc = sc.tolist()
        semantic_std = np.std(sc, ddof=1)

        semantic_mean = np.mean(sc)
        while l < b:
            if (sc[l] - semantic_mean) + weight * semantic_std < 0:
                variable_pos.append(l+1)
            l += 1
        sentence=re.split(r' +',sentences[i])
        for pos in variable_pos[:len(variable_pos)]:
            sentence[pos]='<*>'
        pos=sc.index(max(sc))+1
        template.append(sentence[pos])  # 加空格
        template.append('\n')
        i += 1
    o = 0
    with open('../SaveFiles&Output/Parseresult/'+ output_class +'/' + str(output_class) + str(output_token) + '.csv', 'w') as f:
        while o < len(template):
            f.write(str(template[o]))
            o += 1
        f.close()


def vary_bert(stage, vocab_size,weight):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_attn_pad_mask(seq_q, seq_k):
        batch_size, seq_len = seq_q.size()
        # eq(zero) is PAD token
        pad_attn_mask = seq_q.data.eq(0).unsqueeze(1)  # [batch_size, 1, seq_len]
        return pad_attn_mask.expand(batch_size, seq_len, seq_len)  # [batch_size, seq_len, seq_len]

    def gelu(x):
        """
          Implementation of the gelu activation function.
          For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
          0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
          Also see https://arxiv.org/abs/1606.08415
        """
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    class Embedding(nn.Module):
        def __init__(self):
            super(Embedding, self).__init__()
            self.tok_embed = nn.Embedding(vocab_size, d_model)  # token embedding
            self.pos_embed = nn.Embedding(maxlen, d_model)  # position embedding
            # segment(token type) embedding
            self.norm = nn.LayerNorm(d_model)

        def forward(self, x):
            seq_len = x.size(1)
            pos = torch.arange(seq_len, dtype=torch.long)
            pos = pos.unsqueeze(0).expand_as(x).to(device)  # [seq_len] -> [batch_size, seq_len]
            embedding = self.tok_embed(x.to(device)) + self.pos_embed(pos.to(device))
            return self.norm(embedding)

    class ScaledDotProductAttention(nn.Module):
        def __init__(self):
            super(ScaledDotProductAttention, self).__init__()

        def forward(self, Q, K, V, attn_mask):
            scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(
                d_k)  # scores : [batch_size, n_heads, seq_len, seq_len]
            scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is one.
            attn = nn.Softmax(dim=-1)(scores)
            at = attn.squeeze(dim=0).squeeze(dim=0)
            if stage == "parse":
                return at.detach().cpu().numpy()
            context = torch.matmul(attn, V)
            return context

    class MultiHeadAttention(nn.Module):
        def __init__(self):
            super(MultiHeadAttention, self).__init__()
            self.W_Q = nn.Linear(d_model, d_k * n_heads)
            self.W_K = nn.Linear(d_model, d_k * n_heads)
            self.W_V = nn.Linear(d_model, d_v * n_heads)

        def forward(self, Q, K, V, attn_mask):
            # q: [batch_size, seq_len, d_model], k: [batch_size, seq_len, d_model], v: [batch_size, seq_len, d_model]
            residual, batch_size = Q, Q.size(0)
            # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
            q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2).to(
                device)  # q_s: [batch_size, n_heads, seq_len, d_k]
            k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1, 2).to(
                device)  # k_s: [batch_size, n_heads, seq_len, d_k]
            v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1, 2).to(
                device)  # v_s: [batch_size, n_heads, seq_len, d_v]

            attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1,
                                                      1).to(
                device)  # attn_mask : [batch_size, n_heads, seq_len, seq_len]

            # context: [batch_size, n_heads, seq_len, d_v], attn: [batch_size, n_heads, seq_len, seq_len]
            context = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
            if stage == "parse":
                return context
            context = context.transpose(1, 2).contiguous().view(batch_size, -1,
                                                                n_heads * d_v).to(
                device)  # context: [batch_size, seq_len, n_heads, d_v]
            output = nn.Linear(n_heads * d_v, d_model).to(device)(context).to(device)
            # return nn.LayerNorm(d_model).to(device)(output.to(device)).to(device)
            return nn.LayerNorm(d_model).to(device)(output.to(device) + weight * residual.to(device)).to(
                device)  # output: [batch_size, seq_len, d_model]

    class PoswiseFeedForwardNet(nn.Module):
        def __init__(self):
            super(PoswiseFeedForwardNet, self).__init__()
            self.fc1 = nn.Linear(d_model, d_ff)
            self.fc2 = nn.Linear(d_ff, d_model)

        def forward(self, x):
            # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
            return self.fc2(gelu(self.fc1(x)))

    class EncoderLayer(nn.Module):
        def __init__(self):
            super(EncoderLayer, self).__init__()
            self.enc_self_attn = MultiHeadAttention()
            self.pos_ffn = PoswiseFeedForwardNet()

        def forward(self, enc_inputs, enc_self_attn_mask):
            enc_outputs = self.enc_self_attn(enc_inputs.to(device), enc_inputs.to(device), enc_inputs.to(device),
                                             enc_self_attn_mask.to(device))  # enc_inputs to same Q,K,V
            if stage == "parse":
                return  enc_outputs
            enc_outputs = self.pos_ffn(enc_outputs).to(device)  # enc_outputs: [batch_size, seq_len, d_model]
            return enc_outputs

    class BERT(nn.Module):
        def __init__(self):
            super(BERT, self).__init__()
            self.embedding = Embedding()
            self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
            self.fc = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.Dropout(0.5),
                nn.Tanh(),
            )
            self.classifier = nn.Linear(d_model, 2)
            self.linear = nn.Linear(d_model, d_model)
            self.activ2 = gelu
            # fc2 is shared with embedding layer
            embed_weight = self.embedding.tok_embed.weight
            self.fc2 = nn.Linear(d_model, vocab_size, bias=False)
            self.fc2.weight = embed_weight

        def forward(self, input_ids, masked_pos):
            output = self.embedding(input_ids.to(device)).to(device)  # [bach_size, seq_len, d_model]
            enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids).to(device)  # [batch_size, maxlen, maxlen]
            for layer in self.layers:
                # output: [batch_size, max_len, d_model]
                output = layer(output, enc_self_attn_mask)
                if stage == "parse":
                    return output
            # it will be decided by first token(CLS)
            ###   h_pooled = self.fc(output[:, 0])  # [batch_size, d_model]
            ####  logits_clsf = self.classifier(h_pooled)  # [batch_size, 2] predict isNext

            masked_pos = masked_pos[:, :, None].expand(-1, -1, d_model)  # [batch_size, max_pred, d_model]
            h_masked = torch.gather(output.to(device), 1,
                                    masked_pos.to(
                                        device))  # masking position [batch_size, max_pred, d_model]  位置对齐，将masked的和原本的token对齐
            h_masked = self.activ2(self.linear(h_masked))  # [batch_size, max_pred, d_model]
            logits_lm = self.fc2(h_masked)  # [batch_size, max_pred, vocab_size]     #
            return logits_lm
    return BERT()


class Parsedata(Data.Dataset):
    def __init__(self, input_ids, masked_tokens, masked_pos):
        self.input_ids = input_ids
        self.masked_tokens = masked_tokens
        self.masked_pos = masked_pos
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.masked_tokens[idx], self.masked_pos[idx]
class MyDataSet(Data.Dataset):
    def __init__(self, input_ids, masked_tokens, masked_pos):
        self.input_ids = input_ids
        self.masked_tokens = masked_tokens
        self.masked_pos = masked_pos
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.masked_tokens[idx], self.masked_pos[idx]


import os
import pandas as pd
import re
import numpy as np


def parsing_to_group(contrbution, sentences, type, number, group_stage1,delimeter): #语义贡献分数，日志语句，属于什么系统，长度，一个空词典
    '''
    将相同长度的日志分组
    '''
    i = 0
    lenset = {}
    result = []
    len_t = int(number)
    print(str(number))
    while i < len(contrbution):
        sentence = []
        c = np.argsort(contrbution[i])
        c = list(reversed(c)) #语义贡献得分从大到小排序
        delimeter=''.join(delimeter)
        sentences[i] = re.sub(delimeter, ' ', sentences[i]).lower().split(' ')
        while '' in sentences[i]:
            sentences[i].remove('')
        sentence = sentences[i]

        for l in range(len(c)):
            k=c[0] + 1
            d=sentence[k]
            if (bool(re.search(r"[\d]", sentence[k]))):
                if  sentence[k]!='ssh2':
                  c.append(c.pop(0))
            #if sentence[k] == '"[instance:' or sentence[k] == '[instance':
                #c.append(c.pop(0))
        i += 1
        #if len_t == 11:
            #print(str(sentence[c[0] + 1]))
        output, lenset = get_prelow(sentence, c, lenset, len_t)  ##输出解析结果
        result.append(output)
    li = lenset[len_t]
    n = len(li)
    for i in range(n):
        li = lenset[len_t]
        li = li[i]
        first = li[len(li) - 1]
        a = 0
        for res in result:
            a += 1
            if res[len(res) - 1] == first:
                if a == 0:
                    group_stage1.setdefault(str(len_t) + ',' + str(i), []).append(res)
                else:
                    group_stage1.setdefault(str(len_t) + ',' + str(i), []).append(res)
    return group_stage1


def get_prelow(sentence, c, lenset, len_t):  # 输入待解析的日志语句 sentec     该sentence的语义得分数组c    需要在哪些模板中去判断lenset
    event = []
    if not len_t in lenset:
        lenset.setdefault(len_t, []).append(sentence)
        sentence.append('Lenth' + str(len_t) + 'Event  ' + str(sentence))
        return sentence, lenset
    count = 0
    for i in lenset[len_t]:  # i表示该长度的日志语句有哪些。

        if sentence[c[0] + 1] == i[c[0] + 1]:  # 如果词典中该长度的日志语句只有一类，那么直接进行匹配判断
            count += 1
            event.append(i)
    output, lenset = findtemplate(count, event, sentence, c, lenset, len_t)
    return output, lenset


def findtemplate(count, event, sentence, c, lenset, len_t):
    output = []
    event1 = []
    if count == 1:
        event = event[0]
        sentence.append(str(event[len(event) - 1]))
        output = sentence
    if count == 0:
        lenset[len_t].append(sentence)  # 不符合同一类日志模板
        sentence.append('Lenth' + str(len_t) + 'Event  ' + str(sentence))
        output = sentence
    if count > 1:
        c = c[1:]
        count = 0

        for i in event:  # i表示该长度的日志语句有哪些。
            k=c[0] + 1
            if "10.1.1.1" in sentence:
                k=k-1
            if sentence[k] == i[k]:  # 如果词典中该长度的日志语句只有一类，那么直接进行匹配判断
                count += 1
                event1.append(i)
        output, lenset = findtemplate(count, event1, sentence, c, lenset, len_t)
    return output, lenset
def sentence_process(sentences2,ss,type,delimeter):   #delimeter让空格转变成[PAD]
    sentences = sentences2  #################存储删去源文件索引的日志语句 # filter '.', ',', '?', '!'  re.sub 正则表达式处理数据
    token_list = list()
    for s in sentences:
        if "<1 sec" in s:
            s = re.sub("<1 sec", "00:00", s)
            print('ok')
        delimeter=''.join(delimeter)
        s = re.sub(delimeter, ' ', s).lower()
        if type == 'Thunderbird':
            s = re.sub('name .+\d', 'name <*>', s)
            s = re.sub('spcr', '1', s)
        if type == 'OpenStack':
            s = re.sub('10 [\d ]+10 [\d ]+', '10 1 1 1 ', s)
            s = re.sub('\?.+?= ', 'abc ', s)
            s = re.sub(' json', '', s)
            s = re.sub('8 5', '85', s)
            s = re.sub('11 5.+?us', '15', s)
        if type == "HDFS" or type=="Proxifier"or type=="Windows"or type=="Android":   #有的数据集有（）的解释说明，但是在标注的模板里被去掉了，这里需要预处理
            s = re.sub('\(.*?\)', '', s).split(' ')
        else:
            s = s.split(' ')
        if ss == '[PAD]':
            s = ",".join(s)            ######用逗号分割和原字符对应的子字符
            s=','+s
        else:
            while '' in s:
                s.remove('')
            s = " ".join(s)
        if len(s) == 0:
            continue


        tokenized_text = tokenizer.tokenize(s)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens])
        k = np.squeeze(tokens_tensor.numpy()).tolist()
        token_list.append(k)
    batch = []
    count_index = 0
    while count_index < len(sentences):

        #########tokens_a_index, tokens_b_index = randrange(len(sentences)), randrange(
        ########   len(sentences))  # sample random index in sentences
        tokens_value = token_list[count_index]
        count_index += 1
        ############tokens_a, tokens_b = token_list[tokens_a_index], token_list[tokens_b_index]
        if isinstance(tokens_value,int):
            tokens_value=[tokens_value]
        input_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("[CLS]")) + tokens_value
        ####nput_ids = [word2idx['[CLS]']] + tokens_a + [word2idx['[SEP]']] + tokens_b + [word2idx['[SEP]']]
        #####　segment_ids = [0] * (1 + len(tokens_a) + 1) + [1] * (len(tokens_b) + 1)

        # MASK LM
        #############################  n_pred = min(max_pred,
        #########################################              max(1, int(len(input_ids) * 0.15)))  # 选择一句话中有多少个token要被mask 15 % of tokens in one sentence
        cand_maked_pos = [i for i, token in enumerate(input_ids)
                          if token != tokenizer.convert_tokens_to_ids(
                tokenizer.tokenize("[CLS]"))]  # 排除分隔的CLS和SEP candidate masked position
        for pos in cand_maked_pos[
                   :len(
                       cand_maked_pos)]:  #########################################for pos in cand_maked_pos[:n_pred]:  # 随机打乱后取前n_pred个token做mask
            masked_tokens, masked_pos = [], []
            masked_pos.append(pos)
            masked_tokens.append(input_ids[pos])
            #####masked_tokens.append(input_ids[pos])
            ####if random() < 0.8:  # 80%                    #按Bert论文概率选取mask的方式
            if len(input_ids)>=maxlen:
                input_ids=input_ids[0:maxlen]
            n_pad = maxlen - len(input_ids)
            input_ids.extend([0] * n_pad)  # 填充操作，每个句子的长度是30 余下部分补000000
            batch.append([input_ids, masked_tokens, masked_pos])
            break
    return batch
def group_member_template(sentences, threshold, cou, parse_result,type):
    result = []
    template = sentences[0]
    template = template[1:len(template) - 1]
    tag = 0
    if len(template) == 1:
        tem_dict={}
        for sentence in sentences:
            if not sentence[1] in tem_dict.keys():
                tem_dict[sentence[1]]=cou
                result.append(sentence[1])
                cou += 1
            t_num=tem_dict[sentence[1]]
            sentence = sentence[0:len(sentence) - 1]
            sentence.append(sentence)
            sentence.append('E' + str(t_num))
            parse_result.append(sentence)
        return result
    for i in range(len(template)):
        compare = []
        compare.append(template[i])
        num = 0
        index = 0
        dict = {}
        for sentence in sentences:
            if len(sentence) <= 1:
                continue
            if num >= threshold:
                break
            if not sentence[i + 1] in compare:
                compare.append(sentence[i + 1])
                template[i] = ' <*>'
                num += 1
            dict.setdefault(sentence[i + 1], []).append(index)
            index += 1
        via=0
        for com in compare:
            if (bool(re.search(r"/", com)))or\
                    (bool(re.search(r"\d", com)))\
                    or com =='mask=ffffffff':
                via += 1
            if com =='jun' :
                via=len(compare)
            if com =='en0:'or com =='en0'or \
                    com =="ipv4"or com =='ee0':
                via=via-1

        if via ==len(compare):
                num=0
        if num >= threshold or num == 0:  ##############如果大于threshold或者没有分组，那么这个组也不会在继续细分了。
            continue

        else:
            for com in compare:
                list = dict[com]  ########属于不同分组的记录了下来，单词种类在5个以下的，即被分为5组，（最后一轮分组）
                s = []

                for l in list:
                    s.append(sentences[l])
                template = group_member_template(s, 1, cou, parse_result,type)
                cou+=1
                if len(template) >= 1:
                    for te in template:
                        result.append(te)
                else:
                    result.append(template)
            return result
    if tag == 0:
        for sentence in sentences:
            sentence = sentence[0:len(sentence) - 1]
            sentence.append(template)
            sentence.append('E' + str(cou))
            parse_result.append(sentence)
        cou += 1
    result.append(template)
    return result
def template_Abstraction(group_stage1,type):
    groups = group_stage1.keys()
    group_member = []
    cou = 0
    parse_result = []
    for group in groups:
        group_members = group_stage1[group]
        # sentence_to_template(group_members,len(group_member[0]))
        for mem in group_member_template(group_members, 3, cou, parse_result,type):
            group_member.append(mem)
            cou += 1
    return group_member, parse_result

def get_ParingAccuracy(type,parse_result):
    df_example = pd.read_csv('../logs/'+type+'/'+type+'_2k.log_structured.csv',
                             encoding='UTF-8', header=0)
    correct_count={}
    truth_predict={}
    isinclude={}
    structured = df_example['EventId']
    c = structured[1]
    input_logs=0
    correct_parsed=0
    for id_t in parse_result:
        input_logs += 1
        if id_t[len(id_t) - 1] not in isinclude:
            isinclude[id_t[len(id_t) - 1]] = structured[int(id_t[0])]
        else:
            if not structured[int(id_t[0])] in correct_count:
                #correct_count[structured[int(id_t[0])]] = 0
                d=correct_count[isinclude[id_t[len(id_t) - 1]]]
                correct_count[isinclude[id_t[len(id_t) - 1]]] = 0
                print(str(d)+'这么多条日志被影响到了结果++'+str(id_t[0])+"被错误的合并分类到了模板："+str(isinclude[id_t[len(id_t) - 1]])+"而他本应该属于"+str(structured[int(id_t[0])])+"##该模的板预测为"+id_t[len(id_t) - 1])
                continue
        if not structured[int(id_t[0])] in correct_count :
            correct_count[structured[int(id_t[0])]] = 1
            truth_predict[structured[int(id_t[0])]]=id_t[len(id_t) - 1]
            continue
        if correct_count[structured[int(id_t[0])]] == 0:
            print(str(int(id_t[0]))+"这个日志的解析被之前的错误解析所影响了"+structured[int(id_t[0])]+"该日志模板的内容被错误分类过")
            continue
        if truth_predict[structured[int(id_t[0])]] != id_t[len(id_t) - 1]:
            e=correct_count[structured[int(id_t[0])]]
            correct_count[structured[int(id_t[0])]] = 0          # 这个组一旦出现不符合该组的日志，直接全部清零，并且不允许增加。
            print(str(e)+'这么多条日志被影响到了结果++'+str(id_t[0])+"一个标注的模板中出现了不该属于该模板的日志，已经被这个占用" + truth_predict[structured[int(id_t[0])]] + "而新的预测模板"+id_t[len(id_t) - 1])
            continue
        if correct_count[structured[int(id_t[0])]] != 0:         #被清零过的组不允许继续增加计数。
            correct_count[structured[int(id_t[0])]] +=1
            continue

    for key in correct_count.keys():
        correct_parsed = correct_parsed+correct_count[key]

    print('################ %       '+type+'     % ################')
    print('################ %      Precision     % ################')
    ac=(correct_parsed/(input_logs))
    print('#                        '+str(ac)+'                      #')
    print('######################################################')
    print('#                        ' + str(input_logs) + '                      #')
    print('################ %       '+type+'     % ################')
    print('################ % Parsing Accuracy % ################')
    ac=(correct_parsed/2000)
    print('#                        '+str(ac)+'                      #')
    print('######################################################')
    return ac
def Parse_with_scrub(type, rgex,delimiter):  ###########根据日志format 得到日志的content,根据Choose来选择是否保留带数字和/的参数
        not_real = [".", "_", "-", ":", "/"]
        rgex=''.join(rgex)
        model_path = '../SaveFiles&Output/modelLinux'
        model = vary_bert("parse", vocab_size).to(device)  # 实例化模型
        model.load_state_dict(torch.load(model_path))
        delimiter = ''.join(delimiter)
        lines = open('../logs/'+type+'/'+type+'_2k.log', encoding='UTF-8')
        line = lines.read()
        if type=='Proxifier':
            line=line.replace("<1 sec","00:00")
        lines.close()
        logID=[]
        n = 0
        sentences_pre=[]
        sentences = line.lower().split('\n')
        cluster_group={}
        for sentence in sentences:
            if sentence == '':
                n += 1
                continue
            if re.match(rgex, sentence) != None:
                b = re.match(rgex, sentence).group(1)
                content = re.sub(delimiter, ' ', b)
            else:
                sentence = [str(n) + '<*>']
                logID.append(" ".join(str(sen) for sen in sentence))
                n += 1
                continue
            sent = "[CLS] "
            if type == "HPC":
               content = re.sub('\*\*\*\*', '** ** ', content)
            if type == "HDFS":
                content = re.sub('\(.*?\)', '', content)
            if type == 'OpenStack':
              content = re.sub('10 [\d ]+10 [\d ]+', '10 1 1 1 ', content)
              content = re.sub('\?.+?= ', 'abc ', content)
              content = re.sub(' json', '', content)
              content = re.sub('8 5', '85', content)
              content = re.sub('11 5.+?us', '15', content)
            if type == 'Thunderbird':
                content = re.sub('name .+\d', 'name <*>', content)
                content = re.sub('spcr', '1', content)
            marked_text = sent + content
            sentences_pre.append(marked_text)
            sentence1 = []
            sentence1.append(re.split(' +', marked_text))
            sentence3 = sentence1[0]
            sentence = [str(n) + ' ']
            sentence.extend(sentence3[1:len(sentence3)])
            logID.append(" ".join(str(sen) for sen in sentence))
            lent = len(sentence1[0])
            if lent <= 1:
                n+=1
                continue
            cluster_group.setdefault(lent, []).append(str(n))
            n+=1
        #starttime = timeit.default_timer()
        groups = cluster_group.keys()
        group_stage1 = {}
        for group in groups:

            sentences_group = []
            for i in cluster_group[group]:
                sentences_group.append(logID[int(i)])
            if group == 2:
                for id_t in sentences_group:
                    id_t1 = re.split(' +', id_t)  # 按数据集格式找到该日志的content
                    id_t1.append(['LEN2'])
                    group_stage1.setdefault(str(2), []).append(id_t1)
                continue


            else:
                print("ok")
            batch = sentence_process(sentences_group, 'No',type)
            i = 0
            batch2 = sentence_process(sentences_group, '[PAD]',type)
            fc = []
            sentences = sentences_group
            while i < len(batch):
                sc = []
                input_ids, masked_tokens, masked_pos = batch[i]
                input_ids[masked_pos[0]] = masked_tokens[0]
                attention_score = model(torch.LongTensor([input_ids]),
                                        torch.LongTensor([masked_pos]))
                if len(input_ids)<maxlen:
                    s = input_ids.index(0)
                else:
                    s=maxlen
                c = attention_score[1:s, 1:s]
                semantic_contrbution = c.sum(axis=0)
                input_ids2, masked_tokens2, masked_pos2 = batch2[i]
                beg = 0
                t = 0
                b = 0
                start = input_ids2.index(1010, 0)
                if not 1010 in input_ids2[start + 1:]:  # 该行为空 跳出循环
                    i += 1
                    continue
                while 1010 in input_ids2[beg + 1:]:

                    beg = input_ids2.index(1010, start + 1)
                    if beg - start - 1 != 0:
                        sc.append(sum(semantic_contrbution[start - t - 1:beg - 2 - t]) / (beg - start - 1))
                        b += 1
                    t += 1
                    start = beg

                if len(semantic_contrbution) - beg + t + 1 != 0:
                    sc.append(sum(semantic_contrbution[beg - t - 1:len(semantic_contrbution)]) / (
                            len(semantic_contrbution) - beg + t + 1))
                fc.append(sc)
                i += 1
            group_stage1 = parsing_to_group(fc, sentences, type, group, group_stage1)
        # 中间写代码
        #end = timeit.default_timer()
        #print('STAGE 2: %s Seconds' % (end - starttime))
        template_set, parse_result = template_Abstraction(group_stage1,type)
        with open('../SaveFiles&Output/Parseresult/' + str(type) + '/'+str(type)+'benchmark.csv', 'a+', encoding='UTF-8') as f:
            for st_res in parse_result:
                f.write(str(st_res))
                f.write('\n')
            f.close()
        print('ok') #########################evaulate result#########################################d#
        ac=get_ParingAccuracy(type, parse_result)
        return ac

def semantic_contribution_score(batch,model):
    input_ids, masked_tokens, masked_pos, = zip(*batch)
    input_ids, masked_tokens, masked_pos, = \
        torch.LongTensor(input_ids), torch.LongTensor(masked_tokens), \
        torch.LongTensor(masked_pos),
    loader = Data.DataLoader(Parsedata(input_ids, masked_tokens, masked_pos), batch_size, shuffle=False)
    f = 0
    for input_ids, masked_tokens, masked_pos in loader:
        input_ids, masked_tokens, masked_pos = input_ids.to(device), masked_tokens.to(device), masked_pos.to(device)
        attention_score_batch = model(input_ids,
                                masked_pos)
        if f==0:
            attention_score_whole = attention_score_batch
        else:
             attention_score_whole=np.concatenate((attention_score_whole,attention_score_batch),0)
        f+=1
    attention_score_whole=np.squeeze(attention_score_whole)
    return attention_score_whole

def Parse_with_scrub1(type, data, delimiter):  ###########根据日志format 得到日志的content,根据Choose来选择是否保留带数字和/的参数
    not_real = [".", "_", "-", ":","/",'\'',';',"\"","[","]",")","(","{","}"]   #这些符号在语义提供里不计算语义贡献分数。
    model_path = '../SaveFiles&Output/modelsave/'+type+'/model'+type
    #model_path = '../SaveFiles&Output/modelsave/HDFS/modelHDFS'
    model = vary_bert("parse", vocab_size,weight=0.01).to(device)  # 实例化模型
    model.load_state_dict(torch.load(model_path))
    sentences = data.tolist()
    preprocess_start=datetime.datetime.now()
    batch1 = sentence_process(sentences, 'No',type,delimiter)
    batch3 = sentence_process(sentences, '[PAD]',type,delimiter)
    preprocess_end = datetime.datetime.now()
    print("Preprocess time for "+ type +"is"+str(preprocess_end-preprocess_start))
    Self_start = datetime.datetime.now()

    i=0
    q=0
    fc = []
    batch_num = 0
    while q < len(batch1):
        if q % batch_size==0:
            batch_start = batch_num
            batch_num += batch_size
            i=0
            if len(batch1[batch_start:])<batch_size:
                batch = batch1[batch_start:]
                batch2=batch3[batch_start:]
            else:
                batch = batch1[batch_start:batch_num]
                batch2 = batch3[batch_start:batch_num]
            attention_score_whole = semantic_contribution_score(batch, model)
        attention_score = attention_score_whole[i]
        sc = []
        input_ids, masked_tokens, masked_pos = batch[i]
        input_ids[masked_pos[0]] = masked_tokens[0]
        if 0 not in input_ids:
            s=maxlen
        else:
            s = input_ids.index(0)
        c = attention_score[1:s, 1:s]
        semantic_contrbution = c.sum(axis=0)
        input_ids2, masked_tokens2, masked_pos2 = batch2[i]
        beg = 0
        t = 0
        b = 0
        start = input_ids2.index(1010, 0)
        if not 1010 in input_ids2[start + 1:]:  # 该行为空 跳出循环
            q += 1
            i += 1
            fc.append([0])
            continue
        while 1010 in input_ids2[beg + 1:]:
            beg = input_ids2.index(1010, start + 1)
            if beg - start - 1 != 0:
                real_word = find_delete(input_ids2[start + 1:beg], not_real)
                tok_nu = 0
                sum_new = 0
                for rew in real_word:
                    tok_nu += 1
                    sum_new += semantic_contrbution[start - t - 1 + rew]
                if tok_nu == 0:  # 如果分割出来的只是一个特殊字符，那么直接让他的语义贡献得分归零。
                    tok_nu = 1
                sc.append(sum_new / tok_nu)
                b += 1
            t += 1
            start = beg

        if len(semantic_contrbution) - beg + t + 1 != 0:
            real_word = find_delete(input_ids2[beg - t - 1:len(semantic_contrbution)], not_real)
            tok_nu = 0
            sum_new = 0
            for rew in real_word:
                tok_nu += 1
                sum_new += semantic_contrbution[beg - t - 1 + rew]
            if tok_nu == 0:  # 如果分割出来的只是一个特殊字符，那么直接让他的语义贡献得分归零。
                tok_nu = 1
            sc.append(sum_new / tok_nu)

        fc.append(sc)
        i += 1
        q+=1
    delimiter = ''.join(delimiter)

    logID = []
    n = 0
    sentences_pre = []
    start_time = datetime.datetime.now()
    print("Self-attention based module  =" + str(start_time - Self_start))
    cluster_group = {}
    for sentence in sentences:
        if sentence == '':
            n += 1
            continue
        content=sentence
        content = re.sub(delimiter, ' ', content)

        sent = "[CLS] "
        if type=="HDFS" or type=="Proxifier"or type=="Windows"or type=="Android":
            content = re.sub('\(.*?\)', '', content)
            content = re.sub("<1 sec", "00:00", content)
        if type == "HPC":
            content = re.sub('\*\*\*\*', '** ** ', content)
        if type == 'OpenStack':
            content = re.sub('10 [\d ]+10 [\d ]+', '10 1 1 1 ', content)
            content = re.sub('\?.+= ', '', content)
            content = re.sub(' json', '', content)
            content = re.sub('8 5', '85', content)
            content = re.sub('11 5.+?us', '15', content)
        if type == 'Thunderbird':
            content = re.sub('name .+\d', 'name <*>', content)
            content = re.sub('spcr', '1', content)
        marked_text = sent + content
        sentence1 = []
        sentence1.append(re.split(' +', marked_text))
        sentence3 = sentence1[0]
        sentence = [str(n) + ' ']
        sentence.extend(sentence3[1:len(sentence3)])
        logID.append(" ".join(str(sen) for sen in sentence))
        lent = len(sentence1[0])
        if lent <= 1:
            n += 1
            continue
        cluster_group.setdefault(lent, []).append(str(n))
        n += 1
    # starttime = timeit.default_timer()
    groups = cluster_group.keys()
    group_stage1 = {}
    for group in groups:

        sentences_group = []
        sem_fc=[]
        for i in cluster_group[group]:
            sentences_group.append(logID[int(i)])
            sem_fc.append(fc[int(i)])
        if group == 2:
                    for id_t in sentences_group:
                        id_t = re.sub(delimiter, ' ', id_t).lower().split(' ')
                        while '' in id_t:
                            id_t.remove('')
                        id_t1 = id_t  # 按数据集格式找到该日志的content
                        id_t1.append(['LEN2'])
                        group_stage1.setdefault(str(2), []).append(id_t1)
                    continue


        else:
            print("ok")

        sentences = sentences_group
        group_stage1 = parsing_to_group(sem_fc, sentences, type, group, group_stage1,delimiter)  #语义贡献分数，日志语句，属于什么系统，长度，更新的词典
    # 中间写代码
    # end = timeit.default_timer()
    # print('STAGE 2: %s Seconds' % (end - starttime))
    template_set, parse_result = template_Abstraction(group_stage1, type)
    with open('../SaveFiles&Output/Parseresult/' + str(type) + '/' + str(type) + 'benchmark.csv', 'w',
              encoding='UTF-8') as f:
        for st_res in parse_result:
            f.write(str(st_res))
            f.write('\n')
        f.close()
    print('ok')  #########################evaulate result#########################################d#
    end_time=datetime.datetime.now()
    print("template extraction model =" +str(end_time-start_time))
    ac = get_ParingAccuracy(type, parse_result)
    return ac


def find_delete(list, token_l):
    output = []
    token_list = []
    for tkk in token_l:
        tl = tokenizer.convert_tokens_to_ids(tkk)[0]
        token_list.append(tl)
    i=0
    for tk in list:
        if tk not in token_list:
            output.append(i)
            i+=1
    return output

def makedata2(path,stage,number,delimeter):
    print('=========== we are in make data ===========')
    sentences = path.tolist()
    batch = []
    count=0
    for s in sentences:
        count += 1
        print(str(count))
        delimeter = ''.join(delimeter)
        s = re.sub(delimeter, ' ', s).lower().split(' ')
        while '' in s:
            s.remove('')
        if len(s) == 0:
            continue
        s = "".join(s)
        tokenized_text = tokenizer.tokenize(str(s))
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        non_pad = np.nonzero(indexed_tokens)[0]
        input_ids_origin = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("[CLS]")) + indexed_tokens
        n_pad = maxlen - len(input_ids_origin)
        if n_pad > 0:
            input_ids_origin.extend([0] * n_pad)  # 填充操作，每个句子的长度是30 余下部分补000000
        else:
            input_ids_origin = input_ids_origin[:maxlen]

        vocab_size = len(tokenizer.vocab)
        cand_maked_pos = [token + 1 for i, token in enumerate(non_pad)
                          if indexed_tokens[token] != tokenizer.convert_tokens_to_ids(
                tokenizer.tokenize("[CLS]")) and token < maxlen - 1]
        random.shuffle(cand_maked_pos)
        c = 0
        if stage == 'Embedding':
            batch.append([input_ids_origin, [], []])
            continue
        for pos in cand_maked_pos[
                   :len(
                       cand_maked_pos)]:  #### mask words #########################################for pos in cand_maked_pos[:n_pred]:  # 随机打乱后取前n_pred个token做mask
            masked_tokens, masked_pos = [], []
            masked_pos.append(pos)
            masked_tokens.append(input_ids_origin[pos])
            #####masked_tokens.append(input_ids[pos])
            input_ids_mask = copy.deepcopy(input_ids_origin)
            index = random.randint(0, vocab_size - 1)  # random index in vocabulary
            while index < 4:  # can't involve 'CLS', 'SEP', 'PAD'
                index = random.randint(0, vocab_size - 1)
            input_ids_mask[pos] = index  # replace
            ####if random() < 0.8:  # 80%                    #按Bert论文概率选取mask的方式
            # make mask
            batch.append([input_ids_mask,  masked_tokens, masked_pos])
            c += 1

    return batch

def train2(data,epoch_n,output,type,number,weight):
    batch=data
    input_ids, masked_tokens, masked_pos, = zip(*batch)
    input_ids, masked_tokens, masked_pos, = \
        torch.LongTensor(input_ids), torch.LongTensor(masked_tokens), \
        torch.LongTensor(masked_pos),
    loader = Data.DataLoader(MyDataSet(input_ids, masked_tokens, masked_pos), batch_size, True)
    model = vary_bert("train",vocab_size,weight).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adadelta(model.parameters(), lr=0.5)
    d=0
    for epoch in range(epoch_n):
        d+=1
        print('=================now you are in =================epoch'+ str(d))
        for input_ids, masked_tokens, masked_pos in loader:
            input_ids, masked_tokens, masked_pos = input_ids.to(device), masked_tokens.to(device), masked_pos.to(device)
            logits_lm = model(input_ids, masked_pos).to(device)
            loss_lm = criterion(logits_lm.view(-1, vocab_size),
                                masked_tokens.view(-1)).to(device)  # for masked LM  Tensor.View元素不变，Tensor形状重构，当某一维为-1时，这一维的大小将自动计算。
            loss_lm = (loss_lm.float()).mean().to(device)
            loss = loss_lm.to(device)
            print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 1==0:
          torch.save(model.state_dict(), 'model' + str(output)+ str(epoch)+'weight='+str(weight))
    torch.save(model.state_dict(), 'model'+ str(output)+'weight='+str(weight))

    # Predict mask tokens ans isNext
    print('============== Train finished==================')

class format_log:    # this part of code is from LogPai https://github.com/LogPai

    def __init__(self, log_format, indir='./'):
        self.path = indir
        self.logName = None
        self.df_log = None
        self.log_format = log_format

    def format(self, logName):


        self.logName=logName

        self.load_data()

        return self.df_log





    def generate_logformat_regex(self, logformat):
        """ Function to generate regular expression to split log messages
        """
        headers = []
        splitters = re.split(r'(<[^<>]+>)', logformat)
        regex = ''
        for k in range(len(splitters)):
            if k % 2 == 0:
                splitter = re.sub(' +', '\\\s+', splitters[k])
                regex += splitter
            else:
                header = splitters[k].strip('<').strip('>')
                regex += '(?P<%s>.*?)' % header
                headers.append(header)
        regex = re.compile('^' + regex + '$')
        return headers, regex
    def log_to_dataframe(self, log_file, regex, headers, logformat):
        """ Function to transform log file to dataframe
        """
        log_messages = []
        linecount = 0
        with open(log_file, 'r', encoding='UTF-8') as fin:
            for line in fin.readlines():
                try:
                    match = regex.search(line.strip())
                    message = [match.group(header) for header in headers]
                    log_messages.append(message)
                    linecount += 1
                    if linecount ==10000:
                        break
                except Exception as e:
                    pass
        logdf = pd.DataFrame(log_messages, columns=headers)
        logdf.insert(0, 'LineId', None)
        logdf['LineId'] = [i + 1 for i in range(linecount)]
        return logdf


    def load_data(self):
        headers, regex = self.generate_logformat_regex(self.log_format)
        self.df_log = self.log_to_dataframe(os.path.join(self.path, self.logName), regex, headers, self.log_format)