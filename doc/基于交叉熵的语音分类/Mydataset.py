
import torch
import numpy as np
from pathlib import Path
from torch.utils import data

## 均值方差归一化。 帮助训练效果比较大的提升
def max_minnorm(x):
    mean = np.mean(x)
    std = np.std(x)
    
    x = (x - mean) / ( std + 1e-5)
    
    return x
    

## 下面两个函数 ，实现，将一个二维矩阵 补零到 指定长度。 （补一列一列的零）. 如果超过 指定的seglen，则切掉多余的。
def pad(x, seglen, mode='wrap'):
    pad_len = seglen - x.shape[1]
    y = np.pad(x, ((0,0), (0,pad_len)), mode=mode)
    return y

def segment2d(x, seglen=128):
    '''
    :param x: npy形式的mel [80,L]
    :param seglen: padding长度
    :return: padding mel
    '''
    ## 该函数将melspec [80,len] ，padding到固定长度 seglen
    if x.shape[1] < seglen:
        y = pad(x, seglen)
    elif x.shape[1] == seglen:
        y = x
    else:
        r = np.random.randint(x.shape[1] - seglen) ## r : [0-  (L-128 )]
        y = x[:,r:r+seglen]
    return y



class MeldataSet(data.Dataset):
    def __init__(self,datadir,melspec_len,test_ratio,hps):
        self.datadir = Path(datadir) ## 将一个目录的字符串，变成Path类的对象
        self.melspec_len = melspec_len
        self.test_ratio = test_ratio # 测试集占总集比例
        self.hps = hps
        ### 数据路径默认格式为 ： "D:\语音数据\语音分类数据demo\p225\p225_001.wav"
        ## 其中说话人name 必定在 路径的倒数第二格。
        
        ###  训练数据的样本： [ feat ， label   ]
        
        
        ## 1.提取 说话人映射表。
        self.spk_index_list = [ subdir.name for subdir in self.datadir.glob("*") if subdir.is_dir() ]
        ## self.spk_index_list ：【d,g,w,y,z】
        self.num_class = len(self.spk_index_list) ## 分类的数量

        ## 根据数据集文件下的目录，取出所有的说话人的名字。
        ## 【p225,p226,p227】
        
        ## 分别将每个人的语音划分出验证集、训练集、测试集。
        
        self.train_samples = [ ]
        self.eval_samples = []
        self.test_samples = []

        
        for subspk_dir in [ subdir for subdir in self.datadir.glob("*") if subdir.is_dir() ]:
            ## 读某个说话人的语音路径。进行划分。
            ## .../z/1.npy  p.parts[-2]: z 
            self.spk_samples = [ [p,self.spk_index_list.index(p.parts[-2])]  for p in  subspk_dir.rglob("*.npy") if p.is_file()  ]
            
            test_num = 20

            self.train_samples += self.spk_samples[:-test_num]
            self.eval_samples += self.spk_samples[-test_num*2:-test_num]
            self.test_samples += self.spk_samples[-test_num:]
            ## 
            
        self.train_nums = len(self.train_samples)
        self.eval_nums = len(self.eval_samples) ## evaluation 
        self.test_nums = len(self.test_samples)

        
        
        ## 保存 spkindex list
        ## 存储参数文件本身
        index_listpath = Path(self.hps["experiment_name"]) / "spk_index_list.txt"
        index_listpath.touch()
        f = index_listpath.open('a')
        for i in range(len(self.spk_index_list)):
                f.write(f"{self.spk_index_list[i]}|{i}\n")
       
        print("训练说话人表:",self.spk_index_list)


    def __getitem__(self, idx):## 样本数据的索引 ：  样本 ： （频谱，标签）
        melp,label = self.train_samples[idx] ## 语音的路径
        
        ## 读取频谱
        mel = np.load(melp) ## 计算得梅尔频谱 np.array ## [513, len]
        mel = segment2d(mel,seglen=self.melspec_len) ##截断或者补零，每条语音固定为 self.melspec_len
        
        ## 归一化
        if self.hps["is_norm"]:
            mel = max_minnorm(mel)
        
        label = torch.LongTensor([label])
        
        return mel,label.squeeze_() ## 


    def __len__(self):
        return self.train_nums
    
if __name__ == '__main__':
    hps = {
        
        ## 文件夹相关
        "experiment_name":"e2", # 实验结果放在哪个文件夹下面？(需要你自己修改)
        # 数据相关
        "cls_datadir":"/home/ywh/workspace/code/danzi/loushui_shengdata_amptitue", # 频谱文件夹。(需要你自己修改)
        "spec_seglen":800,
        "sample_rate":16000,
        
        
        ## 训练相关:
        "batch_size":32,
        "total_epoch_num":50, ## 跑的轮数
        "eval_interval":500, # 每隔多少步验证一次，验证准确率和测试准确率。
        "model_save_inter":10, # 模型隔多少步存储一次。
        "is_norm":True,
        
    }
    train_mel_dataset = MeldataSet(datadir="/home/ywh/workspace/code/danzi/loushui_shengdata_amptitue",
                                   
                                   melspec_len=200,
                                   test_ratio=0.05,
                                   hps=hps
                                   )
    for (mel,label) in train_mel_dataset:
        print(mel.shape,label)
    pass
