# -*- coding:utf-8 -*-
# @FileName : frame.py
# @Time : 2024/3/12 19:52
# @Author : fiv


"""
分帧：fn = (N - lframe) / mframe + 1
"""
import numpy as np


def frame(x, lframe, mframe):
    """
    :param x: 输入信号
    :param lframe: 帧长
    :param mframe: 帧移
    :return: 分帧后的信号
    """
    N = len(x)
    fn = (N - lframe) // mframe + 1
    return np.array([x[i * mframe:i * mframe + lframe] for i in range(fn)])

# print(frame([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 3, 3))
# import librosa
# wav, sr = librosa.load('../../data/demo.wav', sr=None)
# print(wav)
# print(frame(wav, 512, 256))


# cn : 预加重 -->> en :
