# Semlog
## Self-Supervised Log Parsing Using Semantic Contribution Difference



![image](https://user-images.githubusercontent.com/84389256/171174096-9937a1f6-e41d-4e84-af17-989db07c9399.png)

Fig.1 Framework of Semlog.

![image](https://user-images.githubusercontent.com/84389256/171174308-c95e6d64-1a3f-42ed-a4a4-e3ad47076311.png)

Fig.2 model structure in Semlog.

### Reproduction
Requirements: python 3.7, pytorch 1.10.1, numpy 1.21.2, scipy 1.7.3, pandas 1.3.5, Cuda 11.3,

1.pip install pytorch_pretrained_bert

2.Run "Semlog/benchmark/Semlog_benchmark.py" to get the results of Semlog directly.

3.Run "logparsers/*.py" to reproduce results of existing parsers

The existing parser code reproduced in this paper relies on [LogPai](https://github.com/logpai).

For the parsers compared in our experiment, we reproduce the code in https://github.com/logpai/logparser.

paper link: https://arxiv.org/pdf/1811.03509.pdf.


