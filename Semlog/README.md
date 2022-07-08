# Semlog
## Self-Supervised Log Parsing Using Semantic Contribution Difference

-Submitted to ISSRE 2022 (The 33rd IEEE International Symposium on Software Reliability Engineering).
Semlog implementation details will be made public after the paper is published.

![image](https://user-images.githubusercontent.com/84389256/171174096-9937a1f6-e41d-4e84-af17-989db07c9399.png)

Fig.1 Framework of Semlog.

![image](https://user-images.githubusercontent.com/84389256/171174308-c95e6d64-1a3f-42ed-a4a4-e3ad47076311.png)

Fig.2 model structure in Semlog.

### Reproduction

Run "Semlog/benchmark/Semlog_benchmark.py" to get the results of Semlog directly.

The existing parser code reproduced in this paper relies on [LogPai](https://github.com/logpai).

For the parsers compared in our experiment, we reproduce the code in https://github.com/logpai/logparser.

paper link: https://arxiv.org/pdf/1811.03509.pdf.

## Results of the experiment in our paper (parsing accuracy)

![image](https://user-images.githubusercontent.com/84389256/171177568-d01f11cc-9c71-462b-a798-d5ad42dd0039.png)

![image](https://user-images.githubusercontent.com/84389256/171178704-0246cacf-de8e-4d11-8b49-a9759f005ed3.png)

## !!!Since Semlog is not designed for batch processing, the parsing process is a bit time consuming (about 10~20 seconds for a Sample dataset).!!!
We have experimented with batch design, the relevant code may be released in the future
