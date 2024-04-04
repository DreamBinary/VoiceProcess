# -*- coding:utf-8 -*-
# @FileName : hmm.py
# @Time : 2024/3/26 20:16
# @Author : fiv
from collections import defaultdict
from pathlib import Path
import os


# 隐马尔科夫链求解词性标注

# pi[q] = 词性q出现所有句子开头的次数 / 所有句子的数量
# trans[q1][q2] = 词性q1后面跟着词性q2的次数 / 词性q1出现的次数
# emit[q][v] = 词性q发射出词v的次数 / 词性q出现的次数

class HMM:
    def __init__(self, corpus_path):
        # self.vocabs, self.classes = self.get_corpus(corpus_path)
        self.corpus_path = corpus_path
        self.line_cnt = 0
        self.states = ['Ag', 'a', 'ad', 'an', 'Bg', 'b', 'c', 'Dg', 'd', 'e', 'f', 'h', 'i', 'j', 'k', 'l', 'Mg', 'm',
                       'Ng', 'n', 'nr', 'ns', 'nt', 'nx', 'nz', 'o', 'p', 'q', 'Rg', 'r', 's', 'na', 'Tg', 't', 'u',
                       'Vg', 'v', 'vd', 'vn', 'vvn', 'w', 'Yg', 'y', 'z']
        self.pi = {state: 0.0 for state in self.states}  # 初始状态概率
        self.trans = {state: {state: 0.0 for state in self.states} for state in self.states}  # 状态转移概率
        self.emit = {state: {} for state in self.states}  # 发射概率
        self.class_cnt = {state: 0 for state in self.states}

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
                self.pi[classes[0]] += 1
                for v, c in zip(vocabs, classes):
                    self.class_cnt[c] += 1
                    if v in self.emit[c]:
                        self.emit[c][v] += 1
                    else:
                        self.emit[c][v] = 1
                for (c1, c2) in zip(classes[:-1], classes[1:]):
                    self.trans[c1][c2] += 1

        self.to_prob()

    def to_prob(self):
        for state in self.states:
            self.pi[state] = self.pi[state] / self.line_cnt
            for e in self.emit[state]:
                self.emit[state][e] = self.emit[state][e] / self.class_cnt[state]
            for t in self.trans[state]:
                self.trans[state][t] = self.trans[state][t] / self.class_cnt[state]

    def viterbi(self, sentence):
        # 初始化
        V = [{}]
        path = {}

        for y in self.states:
            V[0][y] = self.pi[y] * self.emit[y].get(sentence[0], 0)
            path[y] = [y]

        # 递推
        for t in range(1, len(sentence)):
            V.append({})
            newpath = {}

            for y in self.states:
                (prob, state) = max(
                    (V[t - 1][y0] * self.trans[y0].get(y, 0) * self.emit[y].get(sentence[t], 0), y0) for y0 in
                    self.states)
                V[t][y] = prob
                newpath[y] = path[state] + [y]
            path = newpath

        # 终止
        (prob, state) = max((V[len(sentence) - 1][y], y) for y in self.states)
        return prob, path[state]


if __name__ == "__main__":
    hmm = HMM("../../data/corpus.txt")
    test_strs = ["今天 天气 特别 好", "欢迎 大家 的 到来", "请 大家 喝茶", "你 的 名字 是 什么"]
    for s in test_strs:
        ss = s.split(" ")
        p, o = hmm.viterbi(ss)
        print(list(zip(ss, o)))
