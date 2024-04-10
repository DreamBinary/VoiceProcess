# -*- coding:utf-8 -*-
# @FileName : gmm_hmm.py
# @Time : 2024/4/9 19:17
# @Author : fiv
# Isolated word recognition

import numpy as np
import librosa
from hmmlearn.hmm import GMMHMM  # GMMHMM -->> Gaussian Mixture Model Hidden Markov Model
from pathlib import Path

from sklearn.cluster import KMeans
from tqdm import tqdm


class IsolatedWordRecognition:
    def __init__(self, n_components=4, n_mix=3, covariance_type='diag', n_iter=100):
        self.models = []
        self.n_components = n_components
        self.n_mix = n_mix
        self.covariance_type = covariance_type
        self.n_iter = n_iter

    def init_para_hmm(self, X, N_state, N_mix):
        # 初始一定从state=0开始
        pi = np.zeros(N_state)
        pi[0] = 1
        # 当前状态转移概率0.5，下一状态转移概率0.5
        # 进入最后一格状态后不再跳出
        A = np.zeros([N_state, N_state])
        for i in range(N_state - 1):
            A[i, i] = 0.5
            A[i, i + 1] = 0.5
        A[-1, -1] = 1
        feas = X
        len_feas = []
        for fea in feas:
            len_feas.append(np.shape(fea)[0])
        _, D = np.shape(feas[0])
        hmm_means = np.zeros([N_state, N_mix, D])
        hmm_sigmas = np.zeros([N_state, N_mix, D])
        hmm_ws = np.zeros([N_state, N_mix])
        for s in range(N_state):
            sub_fea_collect = []
            for fea, T in zip(feas, len_feas):
                T_s = int(T / N_state) * s
                T_e = (int(T / N_state)) * (s + 1)
                sub_fea_collect.append(fea[T_s:T_e])
            ws, mus, sigmas = self.gen_para_GMM(sub_fea_collect, N_mix)
            hmm_means[s] = mus
            hmm_sigmas[s] = sigmas
            hmm_ws[s] = ws
        return pi, A, hmm_means, hmm_sigmas, hmm_ws

    def gen_para_GMM(self, fea_collect, N_mix):
        # 首先对特征进行kmeans聚类
        feas = np.concatenate(fea_collect, axis=0)
        N, D = np.shape(feas)
        # 初始化聚类中心
        labs = KMeans(n_clusters=N_mix, random_state=9).fit_predict(feas)  # 聚成K类
        mus = np.zeros([N_mix, D])
        sigmas = np.zeros([N_mix, D])
        ws = np.zeros(N_mix)
        for m in range(N_mix):
            index = np.where(labs == m)[0]
            sub_feas = feas[index]
            mu = np.mean(sub_feas, axis=0)
            sigma = np.var(sub_feas, axis=0)
            sigma = sigma + 0.0001
            mus[m] = mu
            sigmas[m] = sigma
            ws[m] = np.shape(index)[0] / N
        ws = (ws + 0.01) / np.sum(ws + 0.01)
        return ws, mus, sigmas

    def train(self, X, len_X, y):

        for i in tqdm(range(len(X)), desc="Train"):
            data = X[i]
            label = y[i]
            pi, A, hmm_means, hmm_sigmas, hmm_ws = self.init_para_hmm(data, self.n_components, self.n_mix)
            model = GMMHMM(n_components=self.n_components, n_mix=self.n_mix, covariance_type=self.covariance_type,
                           n_iter=self.n_iter)
            model.startprob_ = pi
            model.transmat_ = A
            model.weights_ = hmm_ws
            model.means_ = hmm_means
            model.covars_ = hmm_sigmas
            model.fit(np.array(data), np.array(len_X[i]))
            self.models.append(model)

    def evaluate(self, X, len_X, y):
        acc = 0
        for i in tqdm(range(len(X)), desc="Evaluate"):
            scores = []
            for model in self.models:
                score = model.score(X[i])
                scores.append(score)
            pred = np.argmax(scores)
            if pred == y[i]:
                acc += 1
        return acc / len(X)


def extract_feature(file_name):
    # 读取音频文件
    y, sr = librosa.load(file_name)
    # 提取特征
    fea = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12, n_mels=24, n_fft=256, win_length=256, hop_length=80, lifter=12)
    # 进行正则化
    mean = np.mean(fea, axis=1, keepdims=True)
    std = np.std(fea, axis=1, keepdims=True)
    fea = (fea - mean) / std
    # 添加一阶差分
    fea_d = librosa.feature.delta(fea)
    fea = np.concatenate([fea.T, fea_d.T], axis=1)
    return fea


def load_data(path: Path, is_train=True):
    X, len_X, y = [], [], []
    path = path / "train" if is_train else path / "demo"
    if is_train:
        for dir in path.iterdir():
            t_X, t_len_X, t_y = [], [], []
            if dir.is_dir():
                label = dir.name
                for file in dir.iterdir():
                    if file.suffix == ".wav":
                        feature = extract_feature(file)
                        t_X.append(feature)
                        t_len_X.append(len(feature))
                        t_y.append(int(label))
                X.append(t_X)
                len_X.append(t_len_X)
                y.append(t_y)

    else:
        for file in path.iterdir():
            if file.suffix == ".wav":
                feature = extract_feature(file)
                X.append(feature)
                len_X.append(len(feature))
                y.append(int(file.stem) // 7 + 1)

    return X, len_X, y


if __name__ == "__main__":
    path = Path("../../data/isolated_word")
    train_X, train_len_X, train_y = load_data(path, is_train=True)
    test_X, test_len_X, test_y = load_data(path, is_train=False)
    model = IsolatedWordRecognition()
    model.train(train_X, train_len_X, train_y)

    acc = model.evaluate(test_X, test_len_X, test_y)
    print(acc)
