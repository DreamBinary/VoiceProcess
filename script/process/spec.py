# -*- coding:utf-8 -*-
# @FileName : spec.py
# @Time : 2024/3/12 19:39
# @Author : fiv
from pathlib import Path
import numpy as np

from script.HTKFeat import MFCC_HTK
import matplotlib.pyplot as plt


def display_spec(filepath: Path):
    assert filepath.exists()
    mfcc = MFCC_HTK()
    signal = mfcc.load_raw_signal(filepath)
    signal = signal[100:]  # remove the noise
    sig_len = signal.size / 16000

    plt.figure(figsize=(10, 4))
    t = np.linspace(0, sig_len, signal.size)
    plt.plot(t, signal)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Raw Signal')
    plt.show()

# display_spec(Path("../../output/voice.wav"))
