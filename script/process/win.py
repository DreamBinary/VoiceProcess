# -*- coding:utf-8 -*-
# @FileName : win.py
# @Time : 2024/3/12 19:47
# @Author : fiv
import matplotlib.pyplot as plt
import numpy as np
from frame import frame

"""
加窗：将信号分割成若干个帧，然后对每一帧信号进行加窗处理 
 """


def win_frame(frame, win_type):
    """
    :param frame: 一帧信号
    :param win_type: 窗函数类型
    :return: 加窗后的信号
    """
    if win_type == "hanning":
        return frame * np.hanning(frame.size)
    elif win_type == "hamming":
        return frame * np.hamming(frame.size)
    elif win_type == "blackman":
        return frame * np.blackman(frame.size)
    elif win_type == "bartlett":
        return frame * np.bartlett(frame.size)
    else:
        raise ValueError("Unsupported window type")


def win_signal(signal, win_type):
    """
    :param signal: 信号
    :param win_type: 窗函数类型
    :return: 加窗后的信号
    """
    frames = frame(signal, 160, 80)
    win_frames = np.zeros(frames.shape)
    for i in range(frames.shape[0]):
        win_frames[i] = win_frame(frames[i], win_type)
    return win_frames


# signal = np.random.random(100)
# signal_win = win_frame(signal, "hanning")
# plt.figure(figsize=(18, 12))
# t = np.linspace(0, 100, signal.size)
# plt.subplot(2, 1, 1)
# plt.plot(t, signal)
# plt.title('Original Signal')
# plt.subplot(2, 1, 2)
# plt.plot(t, signal_win)
# plt.title('Windowed Signal')
#
# plt.show()

# signal = np.random.random(1000)
# signal_win = win_signal(signal, "hanning")
# print(signal.shape)
# print(signal_win.shape)
