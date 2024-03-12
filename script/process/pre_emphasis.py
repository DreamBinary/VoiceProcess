# -*- coding:utf-8 -*-
# @FileName : pre_emphasis.py
# @Time : 2024/3/12 20:31
# @Author : fiv
import numpy as np

"""
预加重：高频增强 为了弥补语音信号的能量随着频率增加而减小的问题
f(n) = s(n) - α * s(n-1) 
 """


def pre_emphasis(wave_data, weight=0.97):
    """
    :param wave_data: 原始语音信号
    :param pre_emphasis: 预加重系数
    :return: 预加重后的语音信号
    """
    # assert wave_data is np.array
    if isinstance(wave_data, list):
        wave_data = np.array(wave_data)
    wave_data = wave_data[1:] - weight * wave_data[:-1]
    return wave_data


# print(pre_emphasis(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])))

# import librosa
# import matplotlib.pyplot as plt
#
# wav, sr = librosa.load('../../data/demo.wav', sr=None)
#
# plt.figure(figsize=(18, 12))
#
# plt.subplot(2, 1, 1)
# plt.plot(wav)
# plt.title('Original Signal')
#
# plt.subplot(2, 1, 2)
# wav = pre_emphasis(wav)
# plt.plot(wav)
# plt.title('Pre-emphasis Signal')
#
# plt.show()
