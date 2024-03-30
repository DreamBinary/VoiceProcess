
import numpy as np
import random
import torch.nn as nn
from torch.utils import data
from pathlib import Path
import torch
from Model import SpeakerEncoder
from Mydataset import MeldataSet
from Mydataset import segment2d

import torch.nn.functional as F

from Mydataset import max_minnorm

## 2023 03 10 ，dhtz writed。
## 定义训练集 数据结构 16-42行
## 已知数据 采样率 32000, 长度是2秒。、 hop size = 256 的话，2秒钟就有 250 帧。
def print_network( model, name):
    """Print out the network information."""
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print("Model {},the number of parameters: {}".format(name, num_params))


## 该方法将一个字典  ，按kv的顺序，写入一行到 log.txt
def write_line2log(log_dict: dict, filedir, is_write=True,isprint=True):
    '''

    :param log_dict:  字典，存储有每行 的log
    :param filedir:  txt文件路径
    :param is_write:  是否写入
    :param isprint:  是否打印
    :return:
    '''

    strp = ''
    for key, value in log_dict.items():
        witem = '{}'.format(key) + ':{},'.format(value)
        strp += witem
    strp += "\n"
    if is_write:
        with open(filedir, 'a', encoding='utf-8') as f:
            f.write(strp)
    if isprint:
        print(strp)
    pass


def train(hps):
    
    
    # 开始训练
    # 训练之前，开启一些文件。
    ep_filedir = Path(hps["experiment_name"])
    ckpts_dir = ep_filedir / "model_checkpoints"
    log_trainpath = ep_filedir / "train_log.txt"
    log_testpath = ep_filedir / "test_log.txt"

    ep_filedir.mkdir(exist_ok=True)
    ckpts_dir.mkdir(exist_ok=True)
    log_trainpath.touch()
    log_testpath.touch()
    f1 = log_trainpath.open('a')
    f2 = log_testpath.open('a')
    ### 


    # 初始化数据集
    ## 构建训练集
    train_mel_dataset = MeldataSet(datadir=hps["cls_datadir"],
                                   melspec_len=hps["spec_seglen"],
                                   test_ratio=0.05,
                                   hps=hps
                                   )
    num_class  = train_mel_dataset.num_class
    
    
    #输入总文件夹路径、切割长度。这里为什么是 200 请看最上面。
    print("dataset创建")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # 训练设备

    batch_size = hps["batch_size"] ## 一次加载 10条语音训练
    model_save_inter = hps["model_save_inter"] # 训练每隔 M 轮  保存一次模型文件。
    #####################################   上面需要设置一些训练超参数              #######################################
    train_loader = data.DataLoader(train_mel_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # 初始化模型
    model_params_config = {"SpeakerEncoder":
                               {"c_in": 513,
                                "c_h": 128,
                                "c_out": 128,
                                "kernel_size": 5,
                                "bank_size": 8,
                                "bank_scale": 1,
                                "c_bank": 128,
                                "n_conv_blocks": 6,
                                "n_dense_blocks": 6,
                                "subsample": [1, 2, 1, 2, 1, 2],  ## 下采样的主要功能：缩小时间帧
                                "act": 'relu',
                                "dropout_rate": 0,
                                "num_class": num_class}
                           }
    model = SpeakerEncoder(**model_params_config['SpeakerEncoder']).to(device)
    print_network(model,'cls')
    print("model创建")
    ## 分类损失函数
    loss_f = nn.CrossEntropyLoss().to(device)
    ## 优化器
    optimizer = torch.optim.Adam(model.parameters(), 1e-4,
                                 betas=(0.99, 0.999))
    ## 记录一些数值
    epoch_num = 0
    train_iter = 0

    print("开始训练")
    for epoch_num in range(hps["total_epoch_num"]):
        for batch in train_loader:
            model.train()
            train_iter += 1
            #
            mels,labels = batch ## [B , D ,T ]
            mels = mels.to(device)
            labels = labels.to(device)

            pred_prob = model(mels)
            batch_loss = loss_f(pred_prob,labels)
            
            ## torch.max 返回 最大值，和最大值所在的 下标
            pred_index = torch.max(pred_prob, dim=1)[1]  # 求 最大概率的下标
            
            batch_acc = (pred_index == labels).sum() / batch_size # 计算单个batch 的正确率

            ## 更新神经网络的参数
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            losse_curves = {
                            "epoch": epoch_num,
                            "steps": train_iter,
                            "loss": batch_loss.item(),
                            "batch_acc": batch_acc}
            
            write_line2log(losse_curves, log_trainpath, is_write=True,isprint=True)
            
            # 隔着 N 步并计算一次验证集，测试集 正确率
            if train_iter % hps["eval_interval"] == 0:  
                ## (2) 计算验证 正确率
                model.eval()
                eval_right = 0
                for evalsa in train_mel_dataset.eval_samples:
                    evalp,label = evalsa
                    eval_mel = segment2d(np.load(evalp),seglen=hps["spec_seglen"])
                    
                    if hps["is_norm"]:
                        eval_mel = torch.FloatTensor(max_minnorm(eval_mel)).to(device)
                    
                    eval_label = torch.LongTensor([label]).to(device)
                    eval_mel = eval_mel.unsqueeze(0) #[1,80,128]
                    eval_out_prob = model(eval_mel)
                    _, maxflags = torch.max(F.softmax(eval_out_prob, dim=-1),
                                            dim=-1)
                    if maxflags == eval_label:
                        eval_right += 1
                eval_acc = eval_right / train_mel_dataset.eval_nums
                losse_curves["eval_acc"] = eval_acc
                ## (3) 测试准确率
                ## (2) 计算测试 正确率
                model.eval()
                test_right = 0
                for testsa in train_mel_dataset.test_samples:
                    testp,label = testsa
                    test_mel = segment2d(np.load(testp),seglen=hps["spec_seglen"])
                    
                    if hps["is_norm"]:
                        test_mel = torch.FloatTensor(max_minnorm(test_mel)).to(device)
                    
                    test_label = torch.LongTensor([label]).to(device)
                    test_mel = test_mel.unsqueeze(0)  # [1,80,200]
                    test_out_prob = model(test_mel)
                    _, maxflags = torch.max(F.softmax(test_out_prob, dim=-1),
                                            dim=-1)
                    if maxflags == test_label:
                        test_right += 1
                test_acc = test_right / train_mel_dataset.test_nums
                losse_curves["test_acc"] = test_acc

                ### log write
                write_line2log(losse_curves, log_testpath, is_write=True,isprint=True)
                print("-"*30)

        ## 保存模型
        if train_iter % model_save_inter == 0 :
            modelsavep = str( ckpts_dir / "epoch_{:06}.pth".format(epoch_num) )
            torch.save({"model":model.state_dict()},
                        modelsavep)
# 随机数种子固定
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    same_seeds(2000)
    ## 运行这个文件 进行神经网络训练。
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
    train(hps=hps)



    pass




