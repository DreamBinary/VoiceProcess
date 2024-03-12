# -*- coding:utf-8 -*-
# @FileName : record.py
# @Time : 2024/3/12 18:43
# @Author : fiv
import pyaudio
import wave
from tqdm import tqdm


def record(wave_out_path: str, record_second: float):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    wf = wave.open(wave_out_path, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    print("--------------->> recording")
    for _ in tqdm(range(0, int(RATE / CHUNK * record_second))):
        data = stream.read(CHUNK)
        wf.writeframes(data)
    print("--------------->> done recording")
    stream.stop_stream()
    stream.close()
    p.terminate()
    wf.close()
