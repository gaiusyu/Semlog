import seaborn
from matplotlib import pyplot as plt
from pytorch_pretrained_bert import BertTokenizer
import os
import re
import math
import torch
import numpy as np
import random
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import time
time_start = time.time()  # 记录开始时间
# function()   执行的程序
tokenizer = BertTokenizer.from_pretrained('../Bert/bert-base-uncased-vocab.txt')
vocab_size=len(tokenizer.vocab)
maxlen = 512
batch_size = 512
max_pred = 5  # 最大被maksed然后预测的个数 max tokens of prediction
n_layers = 6  # encoder的层数
n_heads = 12  # 多头注意力机制头数
d_model = 768  # 中间层维度
d_ff = 768 * 4  # 全连接层的维度 4*d_model, FeedForward dimension
d_k = d_v = 64  # QKV的维度 dimension of K(=Q), V
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
    sentences1 = line.lower().split('\n')  ################## 包含索引的文件
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
        masked_tokens, masked_pos = [], []

        for pos in cand_maked_pos[
                   :len(
                       cand_maked_pos)]:  #########################################for pos in cand_maked_pos[:n_pred]:  # 随机打乱后取前n_pred个token做mask
            masked_tokens, masked_pos = [], []
            masked_pos.append(pos)
            masked_tokens.append(input_ids[pos])
            origin_input_id = input_ids
            #####masked_tokens.append(input_ids[pos])
            ####if random() < 0.8:  # 80%                    #按Bert论文概率选取mask的方式
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
    time = 3
    t = 0
    sentences = line.lower().split('\n')
    while i < len(batch):
        semantic_contrbution = []
        variable_pos = []
        sc=[]
        clustering = []
        classfication=[]
        n = 1
        l = 0
        m = 0
        p = 1  # [CLS]不会输出到解析结果中
        q = 1
        v = 1
        input_ids, masked_tokens, masked_pos = batch[i]

        print('================================'+str(i))
        input_ids[masked_pos[0]] = masked_tokens[0]
        '''
        while v < len(input_ids):  # 将这条日志语句的模板输出
            if idx2word[input_ids[v]] != '[PAD]':
                template.append(idx2word[input_ids[v]] + ' ')  # 加空格
            v += 1
        '''

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
        a=len(batch)
        b=len(batch2)
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
       # for pos in variable_pos[:len(variable_pos)]:
        #    sentence[pos]='<*>'
          # 加空格
        sc[sc.index(max(sc))]=0
        #sc[sc.index(max(sc))]=0
        sc[sc.index(max(sc))] = 0
        pos = sc.index(max(sc)) + 1
        template.append(sentence[pos])
        template.append('\n')
        i += 1
        '''    #调试代码
        if i == 7:
            break
     
       '''
    o = 0
    with open('../SaveFiles&Output/Parseresult/'+ output_class +'/' + str(output_class) + str(output_token) + '.csv', 'w') as f:
        while o < len(template):
            f.write(str(template[o]))
            o += 1
        f.close()


def vary_bert(stage, vocab_size):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    '''
    loader = Data.DataLoader(MyDataSet(input_ids, masked_tokens, masked_pos), batch_size, True)
    '''

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
            at=torch.sum(at,dim=0)
            #b=at.detach().cpu().numpy()
            #np.savetxt('../SaveFiles&Output/attnscore/Spark.csv', b, delimiter=',')
            #plt.figure(figsize=(40, 20),dpi=400)
            #seaborn.heatmap(b)
            #plt.savefig('../SaveFiles&Output/attngraph/Spark.png')
            if stage == "parse":
                return at.detach().cpu().numpy()
            # torch.set_printoptions(edgeitems=)
            # print('ATTTTTTTTTTTTTTTTTTTTTTTTTTTN',attn)
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
            return nn.LayerNorm(d_model).to(device)(output.to(device) + 0.01 * residual.to(device)).to(
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

def train(path,epoch_n,output,type,number):
    batch=makedata(1,path)
    input_ids, masked_tokens, masked_pos, = zip(*batch)
    input_ids, masked_tokens, masked_pos, = \
        torch.LongTensor(input_ids), torch.LongTensor(masked_tokens), \
        torch.LongTensor(masked_pos),
    loader = Data.DataLoader(MyDataSet(input_ids, masked_tokens, masked_pos), batch_size, True)
    model = vary_bert("train",vocab_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adadelta(model.parameters(), lr=0.01)  # 0.00001
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
            ######## loss_clsf = criterion(logits_clsf, isNext)  # for sentence classification
            ######## loss = loss_lm + loss_clsf
            loss = loss_lm.to(device)
            print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 5==0:
          torch.save(model.state_dict(), '../SaveFiles&Output/modelsave/ANDROIDDDDDDmodel' + str(output) + 'epoch' + str(epoch))
    torch.save(model.state_dict(), '../SaveFiles&Output/modelsave/ANDROIDDDDmodel'+ str(output))

    # Predict mask tokens ans isNext
    print('============== Train finished==================')

class MyDataSet(Data.Dataset):
    def __init__(self, input_ids, masked_tokens, masked_pos):
        self.input_ids = input_ids
        self.masked_tokens = masked_tokens
        self.masked_pos = masked_pos
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.masked_tokens[idx], self.masked_pos[idx]

def makedata(insequence,path):
    print('=========== we are in make data ===========')
    if os.path.exists(path) == True:
        lines = open(path,encoding='UTF-8')
        line = lines.read()
        token_list = []
        sentences = []  #################存储删去源文件索引的日志语句
        sentences1 = re.sub("[.,!?\\-_]", '', line.lower()).split('\n')  ################## 包含索引的文件
        sentences2 = line.lower().split('\n')  # filter '.', ',', '?', '!'  re.sub 正则表达式处理数据
        for sentence in sentences2:
            Index = sentence.find(' ')  # 按数据集格式找到该日志的content
            content = sentence[Index + 1:]
            sentences.append(content)
        for s in sentences:
            tokenized_text = tokenizer.tokenize(s)
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
            b = tokenizer.convert_ids_to_tokens(indexed_tokens)
            tokens_tensor = torch.tensor([indexed_tokens])
            k = np.squeeze(tokens_tensor.numpy()).tolist()
            token_list.append(k)
        vocab_size = len(tokenizer.vocab)
        print('data test')
        lines.close()
        batch = []
        count_index = 0
        while count_index < len(sentences) and insequence == 0:
            #########tokens_a_index, tokens_b_index = randrange(len(sentences)), randrange(
            ########   len(sentences))  # sample random index in sentences
            tokens_value = token_list[count_index]
            count_index += 1
            ############tokens_a, tokens_b = token_list[tokens_a_index], token_list[tokens_b_index]
            input_ids = [tokenizer.convert_tokens_to_ids('[CLS]')] + tokens_value
            ####nput_ids = [word2idx['[CLS]']] + tokens_a + [word2idx['[SEP]']] + tokens_b + [word2idx['[SEP]']]
            #####　segment_ids = [0] * (1 + len(tokens_a) + 1) + [1] * (len(tokens_b) + 1)

            # MASK LM
            #############################  n_pred = min(max_pred,
            #########################################              max(1, int(len(input_ids) * 0.15)))  # 选择一句话中有多少个token要被mask 15 % of tokens in one sentence
            cand_maked_pos = [i for i, token in enumerate(input_ids)
                              if token != tokenizer.convert_tokens_to_ids(
                    '[CLS]')]  # 排除分隔的CLS和SEP candidate masked position
            random.shuffle(cand_maked_pos)  # 随机打乱过后
            masked_tokens, masked_pos = [], []
            n_pad = maxlen - len(input_ids)
            input_ids.extend([0] * n_pad)  # 填充操作，每个句子的长度是30 余下部分补000000
            ######### segment_ids.extend([0] * n_pad)  # 填充操作，每个句子的长度是30 余下部分补000000

            # Zero Padding (100% - 15%) tokens
            ########if max_pred > n_pred:
            ###########n_pad = max_pred - n_pred
            ###########masked_tokens.extend([0] * n_pad)  # masked的token最大5个，这个数组也要补0000，当你mask数量不足时
            ###############masked_pos.extend([0] * n_pad)

            ##########if tokens_a_index + 1 == tokens_b_index and positive < batch_size / 2:
            batch.append([input_ids, masked_tokens, masked_pos])  # IsNext
            ##############positive += 1
            ########elif tokens_a_index + 1 != tokens_b_index and negative < batch_size / 2:
            ################batch.append([input_ids, segment_ids, masked_tokens, masked_pos, False])  # NotNext
            ######### negative += 1
        while count_index < len(sentences) and insequence == 1:

            #########tokens_a_index, tokens_b_index = randrange(len(sentences)), randrange(
            ########   len(sentences))  # sample random index in sentences+
            tokens_value = token_list[count_index]
            count_index += 1
            input_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("[CLS]")) + tokens_value
            ############tokens_a, tokens_b = token_list[tokens_a_index], token_list[tokens_b_index]
            ####nput_ids = [word2idx['[CLS]']] + tokens_a + [word2idx['[SEP]']] + tokens_b + [word2idx['[SEP]']]
            #####　segment_ids = [0] * (1 + len(tokens_a) + 1) + [1] * (len(tokens_b) + 1)

            # MASK LM
            #############################  n_pred = min(max_pred,
            #########################################              max(1, int(len(input_ids) * 0.15)))  # 选择一句话中有多少个token要被mask 15 % of tokens in one sentence
            cand_maked_pos = [i for i, token in enumerate(input_ids)
                              if token != tokenizer.convert_tokens_to_ids(
                    tokenizer.tokenize("[CLS]"))]  # 排除分隔的CLS和SEP candidate masked position
            random.shuffle(cand_maked_pos)  # 随机打乱过后
            masked_tokens, masked_pos = [], []
            c=0
            for pos in cand_maked_pos[
                       :len(
                           cand_maked_pos)]:  #########################################for pos in cand_maked_pos[:n_pred]:  # 随机打乱后取前n_pred个token做mask
                input_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("[CLS]")) + tokens_value
                masked_tokens, masked_pos = [], []
                masked_pos.append(pos)
                masked_tokens.append(input_ids[pos])
                #####masked_tokens.append(input_ids[pos])

                if random.random() < 0.8:  # 80%                    #按Bert论文概率选取mask的方式
                    input_ids[pos] = 103  # make mask
                elif random.random() > 0.9:  # 百分之十的几率随机替换为其他单词 10%
                    index = random.randint(0, vocab_size - 1)  # random index in vocabulary
                    while index < 4:  # can't involve 'CLS', 'SEP', 'PAD'
                        index = random.randint(0, vocab_size - 1)
                    input_ids[pos] = index  # replace
                ####if random() < 0.8:  # 80%                    #按Bert论文概率选取mask的方式
                # make mask
                n_pad = maxlen - len(input_ids)
                input_ids.extend([0] * n_pad)  # 填充操作，每个句子的长度是30 余下部分补000000
                batch.append([input_ids, masked_tokens, masked_pos])
                c+=1
                if c == len(cand_maked_pos):
                   break
        print('test')
        return batch
    else:
       print('dataset file not exist')

def getcontentwithoutnumber(input_path, output_path,rgex,choose):      ###########根据日志format 得到日志的content,根据Choose来选择是否保留带数字和/的参数
    lines = open(input_path, encoding='UTF-8')
    line = lines.read()
    lines.close()
    time = 3
    t = 0
    sentences = re.sub("[,!?=]", ' ', line.lower()).split('\n')
    with open(output_path, 'w', encoding='UTF-8') as f:  # 将日志content输出到一个文件夹
        d=0
        for sentence in sentences:
            print('============= you are processing ====== '+ str(d))
            d+=1
            if sentence == '':
                continue
            if re.match(rgex, sentence) != None:
                b = re.match(rgex, sentence).group(1)
                content = b
            else:
                continue
            start =0
            sent="[CLS] "
            content=re.sub('\(.*?\)','',content)
            if choose=='num':

                marked_text = sent + content

                f.write(marked_text + '\n')
                t += 1
            else:
                while content.find(' ', start) != -1:
                    endpos = content.find(' ', start + 1)
                    if endpos == -1:
                        if (bool(re.search(r"[\d](?![^(]*\))", content[start:])) or bool(
                                re.search(r'/', content[start:]))) == False:
                            sent = sent + content[start:]
                        start = endpos
                        break
                    else:
                        if (bool(re.search(r"[\d](?![^(]*\))", content[start:endpos])) or bool(
                                re.search(r'/', content[start:endpos]))) == False:
                            sent = sent + content[start:endpos]
                        start = endpos

                marked_text = sent

                f.write(marked_text + '\n')
                t += 1


    f.close()

def train_parse(type,cluster_list):
    lines = open(cluster_list, encoding='UTF-8')
    line = lines.read()
    lines.close()
    for number in line.split('\n'):
        if os.path.exists(str('../SaveFiles&Output/Cluster/')+ type + '/'  + str(type) + number + '.csv') == True:
            print(number)
            train(str('../SaveFiles&Output/Cluster/')+ type + '/'  + str(type) + number + '.csv', 20,str(type)+number,str(type),number)
            #parse('../SaveFiles&Output/modelsave/model' + str(type) + number, str('../SaveFiles&Output/Cluster/')+ type + '/'  + str(type) + number + '.csv',type, number)
        else:
            continue

def parse_from_model():
    i = 0
    while i > -1:
        if os.path.exists(str('../SaveFiles&Output/Cluster/') + str(type) + str(i) + '.csv') == True:
            print(i)
            parse('../SaveFiles&Output/modelsave/model'+ str(type) + str(i), str('../SaveFiles&Output/Cluster/') + str(type) + str(i) + '.csv', i)
        else:
            break
        i += 1
def parse_certain_cluster(type,i):
        if os.path.exists(str('../SaveFiles&Output/Cluster/')+ type + '/'  + str(type) + str(i) + '.csv') == True:
            print(i)
            #parse('../SaveFiles&Output/modelsave/model'+ str(type) + str(i), str('../SaveFiles&Output/Cluster/')+ type + '/'  + str(type) + str(i) + '.csv', type, str(i))
            parse('modelsave/Apache/DDDOOPmodelApache999',
                  str('../SaveFiles&Output/Cluster/') + type + '/' + str(type) + str(i) + '.csv', type, str(i))
        else:
            print('cluster' + str(i) + '.csv' + 'dosent exist')


def train_certain_cluster(type,i):
    if os.path.exists(str('../SaveFiles&Output/Cluster/')+ type + '/' +type + str(i) + '.csv') == True:
        print(i)
        train(str('../SaveFiles&Output/Cluster/')+ type + '/' +type + str(i) + '.csv', 30, str(type)+str(i),str(type),str(i))
    else:
        print('cluster' + str(i) + '.csv' + 'dosent exist')
def get_eval_metric(result_path,template_path):
    accurate_event = 0
    lines = open(result_path, encoding='UTF-8')
    line = lines.read()
    lines.close()
    results = line.split('\n')
    lines1 = open(template_path, encoding='UTF-8')
    line1 = lines1.read()
    lines1.close()
    a=len(results)
    for result in results:
        a=result.strip() in line1.strip()
        if a:
           accurate_event+=1
    print('\n === Evaluation Result on ===' + result_path)
    parsing_accuracy=float(accurate_event)/len(results)
    print(parsing_accuracy)

def clusteringwithlen(content_path, output):
        i=0
        n=0
        b2 = open(content_path, encoding='UTF-8')  # 加载embedding
        line2 = b2.read()
        b2.close()
        sentences = line2.split('\n')
        array_len=[]

        for sentence in sentences:
            sentence1=[]
            sentence2=sentence
            sentence1.append(re.split(' +',sentence))
            sentence=sentence1[0]

            lent = len(sentence)
            if lent <= 1:
                continue
            if str(lent) not in array_len:
                array_len.append(str(len(sentence)))
                with open('../SaveFiles&Output/Cluster/' + str(output) + '/array_len.csv', 'a+', encoding='UTF-8') as f:
                    f.write(str(len(sentence))+'\n')
                    f.close()
                i+=1
            f = open('../SaveFiles&Output/Cluster/' + str(output) + '/'+ str(output) + str(len(sentence)) + '.csv',
                     'a+',encoding='UTF-8')  # 使用‘a'来提醒python用附加模式的方式打开
            f.write(sentence2 + '\n')
            f.close()
            n+=1
def get_20P(input_path,type):
    lines = open(input_path)  # 加载embedding
    line = lines.read()
    lines.close()
    sentences = line.split('\n')
    count=int(0.2*len(sentences))
    index=random.sample(range(0,len(sentences)),count)
    with open('../SaveFiles&Output/Cluster/' + type + '/'+type+'20P.csv', 'w') as f:
        for i in index:
            f.write(sentences[i]+'\n')
        f.close()
'''
def parse_realtime(type,i):
    getcontent('../SaveFiles&Output/dataset/HDFS/HDFS2k.log','../SaveFiles&Output/dataset/HDFS/HDFScontent.csv',r'^.+?: (.+)$')
    getcontentwithoutnumber('../SaveFiles&Output/dataset/HDFS/HDFS2k.log', '../SaveFiles&Output/dataset/HDFS/HDFSwithoutnumber.csv',
                            r'^.+?: (.+)$')
    b1 = open(contentwithoutnumber_path, encoding='UTF-8')  # 加载embedding
    line = b1.read()
    b1.close()
    sentences=line.split()
    for sentence in sentences:
        if len(sentence.split()) > 1 and os.path.exists('../SaveFiles&Output/modelsave/model' + str(type) + str(len(sentence.split()))) == True:
            parse('../SaveFiles&Output/modelsave/model' + str(type) + str(len(sentence.split())),
                  str('../SaveFiles&Output/Cluster/') + type + '/' + str(type) + str(len(sentence.split())) + '.csv', type, i) 
            
        f = open('../SaveFiles&Output/Cluster/' + str(output) + '/' + str(output) + str(len(sentence.split())) + '.csv',
                 'a+')  # 使用‘a'来提醒python用附加模式的方式打开
        f.write(sentences2[n] + '\n')
        f.close()
        n += 1
    
    if os.path.exists(str('../SaveFiles&Output/Cluster/') + type + '/' + str(type) + str(i) + '.csv') == True:
        print(i)
        parse('../SaveFiles&Output/modelsave/model' + str(type) + str(i),
              str('../SaveFiles&Output/Cluster/') + type + '/' + str(type) + str(i) + '.csv', type, i)
    else:
        print('cluster' + str(i) + '.csv' + 'dosent exist')


batch, word2idx, idx2word, vocab_size, word_list, token_list=makedata(1,'../SaveFiles&Output/Cluster/HDFS/HDFS3.csv')
np.save('../SaveFiles&Output/Cluster/word2idx.npy',word2idx)
file=open('../SaveFiles&Output/Cluster/token_list.txt','w')
for token in token_list:
    file.write(str(token))
    file.write('\n')
file.close()
np.save('../SaveFiles&Output/Cluster/idx2word.npy',idx2word)
'''
#train_certain_cluster('HDFS',13)
#parse_certain_cluster('HDFS','13')
#train_parse('Proxifier','../SaveFiles&Output/Cluster/Proxifier/array_len.csv')
#clusteringwithlen('../SaveFiles&Output/dataset/HDFS/HDFSwithoutnumber.csv','../SaveFiles&Output/dataset/HDFS/HDFScontent.csv', 'HDFS')
#train_certain_cluster('Proxifier',)
#getcontent('../SaveFiles&Output/dataset/HDFS/HDFS2k.log','../SaveFiles&Output/dataset/HDFS/HDFScontent.csv',r'^.+?: (.+)$')
#getcontentwithoutnumber('../SaveFiles&Output/dataset/HDFS/HDFS2k.log','../SaveFiles&Output/dataset/HDFS/HDFSwithoutnumber.csv',r'^.+?: (.+)$','nonum')
#getcontentwithoutnumber('../SaveFiles&Output/dataset/HDFS/HDFS2k.log','../SaveFiles&Output/dataset/HDFS/HDFScontent.csv',r'^.+?: (.+)$','num')
#get_eval_metric('../SaveFiles&Output/Parseresult/HDFS/HDFS8.csv','../SaveFiles&Output/Parseresult/HDFS/template.csv')

def parse_graph(type, i):
    if os.path.exists(str('../SaveFiles&Output/Cluster/') + type + '/' + str(type) + str(i) + '.csv') == True:
        print(i)
        contrbution,input_ids=acc_parse('../SaveFiles&Output/modelsave/HHHHPCCCCmodelHPC'+str(i),
              str('../SaveFiles&Output/Cluster/') + type + '/' + str(type) + str(i) + '.csv', type, str(i))
        return contrbution,input_ids
    else:
        print('cluster' + str(i) + '.csv' + 'dosent exist')

def acc_parse(model_path, input_path, output_class,
          output_token):  # 输入对应的模型路径，需要解析的文件路径，输出到Parseresult文件夹下，output_class为处理文件的类别，Output_token为属于类别的长度
    model = vary_bert("parse", vocab_size).to(device)  # 实例化模型
    model.load_state_dict(torch.load(model_path))
    batch = file_batch(input_path, 'No')
    i = 0
    weight = 0.2
    template = []
    batch2 = file_batch(input_path, '[PAD]')
    lines = open(input_path, encoding='UTF-8')
    line = lines.read()
    lines.close()
    fc = []

    sentences = re.sub("[.,!?\\-_=:]", ' ', line.lower()).split('\n')
    while i < len(batch):
        semantic_contrbution = []
        sc = []
        n = 1
        l = 0
        m = 0
        p = 1  # [CLS]不会输出到解析结果中
        q = 1
        v = 1
        input_ids, masked_tokens, masked_pos = batch[i]

        print('================================' + str(i))
        input_ids[masked_pos[0]] = masked_tokens[0]
        '''
        while v < len(input_ids):  # 将这条日志语句的模板输出
            if idx2word[input_ids[v]] != '[PAD]':
                template.append(idx2word[input_ids[v]] + ' ')  # 加空格
            v += 1
        '''

        attention_score = model(torch.LongTensor([input_ids]),
                                torch.LongTensor([masked_pos]))
        s = input_ids.index(0)
        c=attention_score[1:s,1:s]

        semantic_contrbution=c.sum(axis=0)
        a = len(batch)
        b = len(batch2)
        input_ids2, masked_tokens2, masked_pos2 = batch2[i]
        beg = 0
        t = 0
        b = 0
        start = input_ids2.index(1010, 0)
        if not 1010 in input_ids2[start + 1:]:    #该行为空 跳出循环
            i+=1
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
        i+=1
    return fc,sentences

###################################################################$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
import os
import pandas as pd
import re
import numpy as np
import test9 as SB


def parsing_to_group(contrbution, sentences, type, number, group_stage1):
    i = 0
    lenset = {}
    result = []
    len_t = int(number)
    while i < len(contrbution):
        sentence = []
        c = np.argsort(contrbution[i])
        c = list(reversed(c))
        event = []
        sentence.append(re.split(' +', sentences[i]))
        sentence = sentence[0]
        for l in range(len(c)):
            if (bool(re.search(r"[\d]", sentence[c[0] + 1]))):
                if  sentence[c[0] + 1]!='ssh2':
                  c.append(c.pop(0))
        i += 1
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
            if sentence[c[0] + 1] == i[c[0] + 1]:  # 如果词典中该长度的日志语句只有一类，那么直接进行匹配判断
                count += 1
                event1.append(i)
        output, lenset = findtemplate(count, event1, sentence, c, lenset, len_t)
    return output, lenset
def sentence_process(sentences2,delimeter):   #delimeter让空格转变成[PAD]
    sentences = []  #################存储删去源文件索引的日志语句 # filter '.', ',', '?', '!'  re.sub 正则表达式处理数据
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
        masked_tokens, masked_pos = [], []

        for pos in cand_maked_pos[
                   :len(
                       cand_maked_pos)]:  #########################################for pos in cand_maked_pos[:n_pred]:  # 随机打乱后取前n_pred个token做mask
            masked_tokens, masked_pos = [], []
            masked_pos.append(pos)
            masked_tokens.append(input_ids[pos])
            origin_input_id = input_ids
            #####masked_tokens.append(input_ids[pos])
            ####if random() < 0.8:  # 80%                    #按Bert论文概率选取mask的方式
            n_pad = maxlen - len(input_ids)
            input_ids.extend([0] * n_pad)  # 填充操作，每个句子的长度是30 余下部分补000000
            batch.append([input_ids, masked_tokens, masked_pos])
            break
    return batch
def group_member_template(sentences, threshold, cou, parse_result):
    result = []
    template = sentences[0]
    template = template[1:len(template) - 1]
    tag = 0
    if len(template) == 1:
        tem_dict={}
        for sentence in sentences:
            if not sentence[1] in tem_dict.keys():
                cou += 1
                tem_dict[sentence[1]]=cou
                result.append(sentence[1])
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
            if (bool(re.search(r"[\d](?![^(]*\))", com))):
                via += 1
            if com =='jun':
                via=len(compare)
            if com =='en0.'or com =='en0' :
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
                template = group_member_template(s, 5, cou, parse_result)
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
def template_Abstraction(group_stage1):
    groups = group_stage1.keys()
    group_member = []
    cou = 0
    parse_result = []
    for group in groups:
        group_members = group_stage1[group]
        # sentence_to_template(group_members,len(group_member[0]))
        for mem in group_member_template(group_members, 5, cou, parse_result):
            group_member.append(mem)
            cou += 1
    return group_member, parse_result

def get_ParingAccuracy(type,parse_result):
    df_example = pd.read_csv('../Log2K_logPai/'+type+'/'+type+'_2k.log_structured.csv',
                             encoding='UTF-8', header=0)
    correct_count={}
    template_num={}
    isinclude={}
    structured = df_example['EventId']
    c = structured[1]
    input_logs=0
    correct_parsed=0
    for id_t in parse_result:
        input_logs += 1
        if not id_t[len(id_t) - 1] in isinclude.keys():
            template_num[id_t[len(id_t) - 1]] = structured[int(id_t[0])]
            if structured[int(id_t[0])]=='E25':
                print('E25 has been taken up by'+str(id_t[0]))
            if structured[int(id_t[0])]=='E15':
                print('E15 has been taken up by'+str(id_t[0]))
            isinclude[id_t[len(id_t) - 1]]=structured[int(id_t[0])]
            if not structured[int(id_t[0])] in correct_count.keys():
               correct_count[structured[int(id_t[0])]]=1

            else:
                correct_count[structured[int(id_t[0])]] = 0
                print("Parsing wrong" + id_t[0] + 'should not be' +
                      structured[int(id_t[0])])
            continue
        if template_num[id_t[len(id_t) - 1]] != structured[int(id_t[0])]:
            correct_count[structured[int(id_t[0])]] = 0
            correct_count[template_num[id_t[len(id_t) - 1]]] = 0     ### Parsing Accuracy, if one log was Parsed wrong , the correct count of two group (it should belong to and it is parsed to) should be 0
            print("Parsing wrong"+id_t[0]+"wrong to"+template_num[id_t[len(id_t) - 1]]+'should be'+structured[int(id_t[0])])
            continue
        else:
            correct_count[structured[int(id_t[0])]] += 1
            continue

    for key in correct_count.keys():
        correct_parsed = correct_parsed+correct_count[key]
    print('################ % Parsing Accuracy % ################')
    ac=(correct_parsed/(input_logs))
    print('#                        '+str(ac)+'                      #')
    print('######################################################')
    print('#                        ' + str(input_logs) + '                      #')
    return 0
def Parse_with_scrub(type, rgex,delimiter):  ###########根据日志format 得到日志的content,根据Choose来选择是否保留带数字和/的参数
        lines = open('../Log2K_logPai/'+type+'/'+type+'_2k.log', encoding='UTF-8')
        line = lines.read()
        if type=='Proxifier':
            line=line.replace("<1 sec","00:00")
        lines.close()
        logID=[]
        n = 0
        sentences_pre=[]
        sentences = line.lower().split('\n')
        cluster_group={}
        test_s=[]
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
            start = 0
            sent = "[CLS] "
            content = re.sub('\(.*?\)', '', content)
            if type == 'OpenStack':
                content = re.sub('10\\..+? 10\\..+? ', '10.1.1.1 ', content)
                content = re.sub('get .+ http/', 'get /v2/54fadb412c4e40cdbaed9335e4c35a9e/servers HTTP/1.1 http',
                                 content)
            marked_text = sent + content
            sentences_pre.append(marked_text)
            sentence1 = []
            sentence2 = marked_text
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
        groups = cluster_group.keys()
        group_stage1 = {}
        for group in groups:
            sentences_group = []
            for i in cluster_group[group]:
                sentences_group.append(logID[int(i)])
            if type != 'Proxifier':
                if os.path.exists('../SaveFiles&Output/modelsave/' + type + '/model' + type + str(group)) == False:
                    if group == 2:
                        for id_t in sentences_group:
                            id_t1 = re.split(' +', id_t)  # 按数据集格式找到该日志的content
                            id_t1.append(['LEN2'])
                            group_stage1.setdefault(str(2), []).append(id_t1)
                    continue

                model_path = '../SaveFiles&Output/modelsave/' + type + '/model' + type + str(
                    group)  ###########################模型位置

            else:
                model_path = '/modelsave/Proxifier/modelProxifier2'
            model = vary_bert("parse", vocab_size).to(device)  # 实例化模型
            model.load_state_dict(torch.load(model_path))
            batch = sentence_process(sentences_group, 'No')
            i = 0
            batch2 = sentence_process(sentences_group, '[PAD]')
            fc = []
            sentences = sentences_group

            input_ids, masked_tokens, masked_pos, = zip(*batch)
            input_ids, masked_tokens, masked_pos, = \
                torch.LongTensor(input_ids), torch.LongTensor(masked_tokens), \
                torch.LongTensor(masked_pos),
            loader = Data.DataLoader(MyDataSet(input_ids, masked_tokens, masked_pos), batch_size, shuffle=False)
            cc=0
            for input_ids, masked_tokens, masked_pos in loader:
                input_ids, masked_tokens, masked_pos = input_ids.to(device), masked_tokens.to(device), masked_pos.to(
                    device)
                attention_score1 = model(input_ids, masked_tokens)
                if cc>=1:
                  attention_score2=np.concatenate((attention_score2,attention_score1), axis=0)
                if cc==0:
                  attention_score2=attention_score1
                cc += 1
            while i < len(batch):
                semantic_contrbution = []
                sc = []
                input_ids, masked_tokens, masked_pos = batch[i]
                input_ids[masked_pos[0]] = masked_tokens[0]
                attention_score2=np.squeeze(attention_score2)
                if attention_score2.ndim != 2:
                  attention_score = attention_score2[i]
                else:
                    attention_score = attention_score2
                s_len = input_ids.index(0)
                c = attention_score[1:s_len, 1:s_len]

                semantic_contrbution = c.sum(axis=0)
                a = len(batch)
                b = len(batch2)
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
        print('get senmatic contribution finished')
        template_set, parse_result = template_Abstraction(group_stage1)
        with open('../SaveFiles&Output/Parseresult/' + str(type) + '/'+str(type)+'benchmark.csv', 'a+', encoding='UTF-8') as f:
            for st_res in parse_result:
                f.write(str(st_res))
                f.write('\n')
            f.close()
        time_end = time.time()  # 记录结束时间
        time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
        print('############ Time taken :' + str(time_sum) + '#################')
        get_ParingAccuracy(type, parse_result)
        return 0
class Parsedata(Data.Dataset):
    def __init__(self, input_ids, masked_tokens, masked_pos):
        self.input_ids = input_ids
        self.masked_tokens = masked_tokens
        self.masked_pos = masked_pos
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.masked_tokens[idx], self.masked_pos[idx]


def evaluate_parse(parse_result, type):
    lines = open('../log2K_logPai/' + str(type) + '/' + str(type) + '_2k.log_structured.csv', encoding='UTF-8')
    line = lines.read()
    lines.close()
    df_example = pd.read_csv('../log2K_logPai/' + str(type) + '/' + str(type) + '_2k.log_structured.csv',
                             encoding='UTF-8')
    # 等同于：
    df_example = pd.read_csv('../log2K_logPai/' + str(type) + '/' + str(type) + '_2k.log_structured.csv',
                             encoding='UTF-8', header=0)
    structured = df_example['EventId']
    c = structured[1]
    print('e')








def template_generate(type):
    counts = open('../SaveFiles&Output/Cluster/' + str(type) + '/array_len.csv', encoding='UTF-8')
    count = counts.read()
    counts.close()
    for number in count.split('\n'):
        get_template(type, int(number))


def get_template(type, lenth):
    count = 0
    while True:
        if os.path.exists('../SaveFiles&Output/Parseresult/' + type + '/CCCCCCLEN' + str(lenth) + str(type) + 'result' + str(
                count) + '.csv') == True:
            lines = open(
                '../SaveFiles&Output/Parseresult/' + type + '/CCCCCCLEN' + str(lenth) + str(type) + 'result' + str(count) + '.csv',
                encoding='UTF-8')
            line = lines.read()
            lines.close()
            sentences = line.split('\n')
            template = sentence_to_template(sentences, lenth)

            with open('../SaveFiles&Output/Parseresult/' + type + '/candidateTemplate.csv', 'a+', encoding='UTF-8') as f:
                for t in template:
                    f.write(str(t) + 'LENTH' + str(lenth) + 'NUM' + str(count) + '\n')
                    print(t)
                f.close()
            count += 1
        else:
            break


def sentence_to_template(sentences, lenth):
    result = []
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
                template = sentence_to_template(s, lenth)
                if len(template) >= 1:
                    for te in template:
                        result.append(te)
                else:
                    result.append(template)
            return result
    result.append(template)
    return result

'''

    for item in list1:
        print("%s has occured for %d" % (str(item), list2.count(item)))
    print('ok')
    '''
print('Hello')
#Parse_with_scrub('HPC',r'^.+? .+? .+? .+? .+? .+? (.+)$',"[,!?\\-_=]")
#Parse_with_scrub('Apache',r'^.+?\] .+?\] (.+)$',"[,!?\\-_=]")
#Parse_with_scrub('Zookeeper',r'^.+?\] - (.+)$',"[,!?\\-_=:]")
#Parse_with_scrub('Mac',r'^.+?\]: (.+)$',"[,!?=:]")
#Parse_with_scrub('HealthApp',r'^.+\|(.+)$',"[,!?=:]")
#Parse_with_scrub('Proxifier',r'^.+?\- (.+)$',"[,!?=]")
#Parse_with_scrub('Android',r'^.+?: (.+)$',"[,!?\\-_=:]")
#Parse_with_scrub('Linux',r'^.+?: (.+)$',"[,!?=:]")



'''
            input_ids, masked_tokens, masked_pos, = zip(*batch)
            input_ids, masked_tokens, masked_pos, = \
                torch.LongTensor(input_ids), torch.LongTensor(masked_tokens), \
                torch.LongTensor(masked_pos),
            loader = Data.DataLoader(MyDataSet(input_ids, masked_tokens, masked_pos), batch_size, True)
            cc=0
            for input_ids, masked_tokens, masked_pos in loader:
                input_ids, masked_tokens, masked_pos = input_ids.to(device), masked_tokens.to(device), masked_pos.to(
                    device)
                attention_score1 = model(input_ids, masked_tokens)
                if cc>=1:
                  attention_score2=np.concatenate((attention_score2,attention_score1), axis=0)
                if cc==0:
                  attention_score2=attention_score1
                cc += 1
            while i < len(batch):
                semantic_contrbution = []
                sc = []
                input_ids, masked_tokens, masked_pos = batch[i]

                print('================================' + str(i))
                input_ids[masked_pos[0]] = masked_tokens[0]
                attention_score = attention_score2[i][0]
                s = input_ids.index(0)
                c = attention_score[1:s, 1:s]

                semantic_contrbution = c.sum(axis=0)
                a = len(batch)
                b = len(batch2)
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
'''