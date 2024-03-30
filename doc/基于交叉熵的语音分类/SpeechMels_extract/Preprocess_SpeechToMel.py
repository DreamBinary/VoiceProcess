
import numpy as np

from pathlib import Path
import librosa

def extract_mel(wav_datadir,output_dir):
    '''
    提取语音.wav数据集的特征
    :param wav_datadir:
    :param output_dir:

    :return:
    '''

    src_wavp = Path(wav_datadir)
    for x in src_wavp.rglob('*'):
        if x.is_dir():
            Path(str(x.resolve()).replace(wav_datadir, output_dir)).mkdir(parents=True, exist_ok=True)
    print("创建相同目录,开始提取特征")

    # 提取特征
    wavpaths = [x for x in src_wavp.rglob('*.wav') if x.is_file()]

    ttsum = len(wavpaths) # 总语音数量
    k = 0
    for wp in wavpaths:
        k += 1
        the_wavpath = str(wp)
        the_melpath = str(wp).replace(wav_datadir, output_dir).replace('.wav', '.npy')


        # 计算log FBank特征
        y, sr = librosa.load(the_wavpath, sr=16000)  # sr=None表示保持原始采样率
        f_bank = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256,
                                            n_mels=80, fmin=0, fmax=8000)

        # 转换为对数刻度
        mel = librosa.power_to_db(f_bank, ref=np.max)
        

        print(f"mel shape:{mel.shape},{k}|{ttsum}")

        np.save(the_melpath,mel)


def wav_to_mel(wavpath):# 计算log FBank特征
     # 计算log FBank特征
    y, sr = librosa.load(wavpath, sr=32000)  # sr=None表示保持原始采样率
    amp =  librosa.stft(y,n_fft=1024,hop_length=256, win_length=1024)
    # 转换为对数刻度
    amp = librosa.power_to_db(amp, ref=np.max) ## -8 到1 之间
    return amp

    


def extract_amptitude(wav_datadir,output_dir):
    '''
    提取语音.wav数据集的特征
    :param wav_datadir:
    :param output_dir:

    :return:
    '''

    src_wavp = Path(wav_datadir) # Path
    
    ## 创建相同结构的新文件目录,开始提取特征,存储到新的目录下。
    for x in src_wavp.rglob('*'):
        if x.is_dir():
            Path(str(x.resolve()).replace(wav_datadir, output_dir)).mkdir(parents=True, exist_ok=True)
    print("创建相同目录,开始提取特征")

    # 提取特征
    wavpaths = [x for x in src_wavp.rglob('*.wav') if x.is_file()]

    ttsum = len(wavpaths) # 总语音数量
    k = 0
    for wp in wavpaths:
        k += 1
        the_wavpath = str(wp)
        the_melpath = str(wp).replace(wav_datadir, output_dir).replace('.wav', '.npy')


        # 计算log FBank特征
        y, sr = librosa.load(the_wavpath, sr=32000)  # sr=None表示保持原始采样率
        amp =  librosa.stft(y,n_fft=1024,hop_length=256, win_length=1024)

        
        # 转换为对数刻度
        amp = librosa.power_to_db(amp, ref=np.max)
        
        print(np.max(amp),np.min(amp))
        print(f"amp shape:{amp.shape},{k}|{ttsum}")

        np.save(the_melpath,amp)
    print("提取完毕。")

if __name__ == '__main__':
    ## 提取 hifigan的melspec

    ## wav 文件的 数据集的目录
    src = '/home/ywh/workspace/code/danzi/loushui_shengdata'

    ## melspec 文件夹
    tar = '/home/ywh/workspace/code/danzi/loushui_shengdata_amptitue'

    extract_amptitude(src,tar)





















