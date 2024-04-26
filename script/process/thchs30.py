# -*- coding:utf-8 -*-
# @FileName : thchs30.py
# @Time : 2024/4/16 19:17
# @Author : fiv
from python_speech_features import logfbank
import librosa


# process thchs30 dataset
class THCHS30:

    def __init__(self, data_path, data_length=None, batch_size=32, shuffle=True):
        self.data_path = data_path
        self.data_length = data_length
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.wav_list = []
        self.pin_list = []
        self.han_list = []
        self.load_data()

    def load_data(self):
        with open(self.data_path, "r") as f:
            for line in f:
                wav, pin, han = line.strip().split("\t")
                self.wav_list.append(wav)
                self.pin_list.append(pin)
                self.han_list.append(han)

        if self.data_length:
            self.wav_list = self.wav_list[:self.data_length]
            self.pin_list = self.pin_list[:self.data_length]
            self.han_list = self.han_list[:self.data_length]

        self.acoustic_vocab = self.acoustic_model_vocab(self.pin_list)
        self.pin_vocab = self.language_model_pin_vocab(self.pin_list)
        self.han_vocab = self.language_model_han_vocab(self.han_list)

    def acoustic_model_vocab(self, data):
        """
        用于确定词汇表的大小
        :param data:
        :return:
        """
        vocab = []
        for line in data:
            line = line
            for pin in line:
                if pin not in vocab:
                    vocab.append(pin)
        vocab.append('_')
        return vocab

    def language_model_pin_vocab(self, data):
        """
        :param data:
        :return:
        """
        vocab = ['<PAD>']
        for line in data:
            for pin in line:
                if pin not in vocab:
                    vocab.append(pin)
        return vocab

    def language_model_han_vocab(self, data):
        vocab = ['<PAD>']
        for line in data:
            line = ''.join(line.split(' '))
            for han in line:
                if han not in vocab:
                    vocab.append(han)
        return vocab

    def ctc_len(self, label):
        add_len = 0

        label_len = len(label)
        for i in range(label_len - 1):
            if label[i] == label[i + 1]:
                add_len += 1
        return label_len + add_len


def to_fbank(wav_path):
    wav, sr = librosa.load(wav_path)
    fbank = logfbank(wav, sr)
    return fbank


if __name__ == '__main__':
    fbank = to_fbank("../../data/demo.wav")
    import matplotlib.pyplot as plt

    plt.imshow(fbank.T)
    plt.show()
