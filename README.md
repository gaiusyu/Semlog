# Semlog
## Self-Supervised Log Parsing Using Semantic Contribution Difference



### Reproduction


1.pip install -r requirements.txt

2.Run "Semlog/benchmark/Semlog_benchmark.py" to get the results of Semlog directly.

3.Run "logparsers/*.py" to reproduce results of existing parsers

#### Docker image:
1. docker pull docker.io/gaiusyu/semlog:v1
2. docker run -it --gpus '"device=0"' --name semlog semlog:v1


The existing parser code reproduced in this paper relies on [LogPai](https://github.com/logpai).

For the parsers compared in our experiment, we reproduce the code in https://github.com/logpai/logparser.

paper link: https://arxiv.org/pdf/1811.03509.pdf.

### Experimental results
![image](https://user-images.githubusercontent.com/84389256/183889194-605d2726-b68b-450c-bfc6-522127874195.png)


