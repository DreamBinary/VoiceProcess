# -*- coding:utf-8 -*-
# @FileName : crf.py
# @Time : 2024/4/4 13:13
# @Author : fiv


from collections import defaultdict
from pathlib import Path
import os
from sklearn_crfsuite import CRF


# 基于条件随机场的词性标注

class CRF_POS:
    def __init__(self, corpus_path):
        self.corpus_path = corpus_path
        self.line_cnt = 0
        self.states = ['Ag', 'a', 'ad', 'an', 'Bg', 'b', 'c', 'Dg', 'd', 'e', 'f', 'h', 'i', 'j', 'k', 'l', 'Mg', 'm',
                       'Ng', 'n', 'nr', 'ns', 'nt', 'nx', 'nz', 'o', 'p', 'q', 'Rg', 'r', 's', 'na', 'Tg', 't', 'u',
                       'Vg', 'v', 'vd', 'vn', 'vvn', 'w', 'Yg', 'y', 'z']
        self.X = []
        self.y = []
        self.crf = CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True,
            verbose=True
        )
        # * ``'lbfgs'`` - Gradient descent using the L-BFGS method   -->> 梯度下降
        # * ``'l2sgd'`` - Stochastic Gradient Descent with L2 regularization term  -->> 随机梯度下降
        # * ``'ap'`` - Averaged Perceptron  -->> 感知机
        # * ``'pa'`` - Passive Aggressive (PA)  -->> 消极攻击
        # * ``'arow'`` - Adaptive Regularization Of Weight Vector (AROW)  -->> 自适应正则化权重向量
        self.train()

    def train(self):
        with open(self.corpus_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines if line.strip()]
            self.line_cnt = len(lines)
            for line in lines:
                vocabs, classes = [], []
                words = line.split(" ")
                for word in words:
                    word = word.strip()
                    if '/' not in word:
                        continue
                    pos = word.index("/")
                    if '[' in word and ']' in word:
                        vocabs.append(word[1:pos])
                        classes.append(word[pos + 1:-1])
                        break
                    if '[' in word:
                        vocabs.append(word[1:pos])
                        classes.append(word[pos + 1:])
                        break
                    if ']' in word:
                        vocabs.append(word[:pos])
                        classes.append(word[pos + 1:-1])
                        break
                    vocabs.append(word[:pos])
                    classes.append(word[pos + 1:])
                assert len(vocabs) == len(classes)
                self.X.append(vocabs)
                self.y.append(classes)
        self.crf.fit(self.X, self.y)

    def predict(self, sentence):
        vocabs = sentence.split(" ")
        return self.crf.predict([vocabs])


if __name__ == '__main__':
    crf = CRF_POS("../../data/corpus.txt")
    test_strs = ["今天 天气 特别 好", "欢迎 大家 的 到来", "请 大家 喝茶", "你 的 名字 是 什么"]
    for test_str in test_strs:
        print(list(zip(test_str.split(" "), crf.predict(test_str)[0])))
