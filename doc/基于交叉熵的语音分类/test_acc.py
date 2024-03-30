import torch
from Model import SpeakerEncoder
from pathlib import Path

from Mydataset import max_minnorm,segment2d
from SpeechMels_extract.Preprocess_SpeechToMel import wav_to_mel




def test_acc(modelpath,evaldatadir,spk_indexlist_path,result_txt_path):
    """_summary_
    该函数利用训练好的模型，测试未知的语音的 漏水声音 类别
    Args:
        modelpath (_type_): 神经网络模型的路径
        evaldatadir (_type_): 验证数据的文件，里面存放.wav文件
        spk_indexlist_path (_type_): 说话人列表文件。
        result_txt_path (_type_): 识别结果输出的文件。
    """
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # 训练设备
    
    
    # 加载说话人表
    spk_indexlist = []
    with open(spk_indexlist_path, 'r') as f2:
        for line  in f2.readlines():
            spkname = line.split("|")[0]
            spk_indexlist.append(spkname)
    num_class = len(spk_indexlist)
    
    ## 加载神经网络模型 
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

    model = SpeakerEncoder(**model_params_config['SpeakerEncoder'])  ## 初始化一个模型 
    model.load_state_dict(torch.load(modelpath)["model"]) ## 模型 加载上 你训练过的模型的参数
    model = model.to(device)

    ## 读取未知类别的.wav文件。
    evaldatadir = Path(evaldatadir)
    evalwavpaths = [ x for x in evaldatadir.rglob('*.wav') if x.is_file()]
    eval_total_num = len(evalwavpaths)
    
    #
    acc_e = 0
    
    with open(result_txt_path,'a',encoding="utf-8") as f1:
        for p in evalwavpaths:
            mel = wav_to_mel(p)
            mel = torch.FloatTensor(max_minnorm(segment2d(mel,seglen=800))).unsqueeze(0).to(device)
            
            pred_prob = model(mel)

            pred_index = torch.max(pred_prob, dim=1)[1]  # 求 最大概率的下标
            
            ## 下标转化为 类别
            pred_class = spk_indexlist[pred_index]## 模型预测
            
            ## 真实类别： 文件名的最后一个字母
            
            
            s = f"测试文件:{str(p)},模型预测类别：{pred_class}" 
            
            
            print(s)
            f1.write(s+"\n")

def test():
    
    ### 测试未知类别的语音。 填好下面四个参数
    
    ## windows电脑的路径格式与linux不一样， 一般是 path  = r="C:\A\B\C"
    
    ## 模型文件的路径
    model_path = "/home/ywh/workspace/code/danzi/语音分类通用项目/e2/model_checkpoints/epoch_000049.pth"
    
    ## 测试语音文件夹 路径
    eval_datadir = "测试语音文件"
    
    #训练时保存的说话人映射表(.txt文件的路径)
    spktxt_path = "e2/spk_index_list.txt"
    
    ## 结果打印的路径。
    outpath = "./test_result.txt"
    
    test_acc(modelpath=model_path,evaldatadir=eval_datadir,spk_indexlist_path=spktxt_path,result_txt_path=outpath)
    

if __name__ == '__main__':

    ## 运行test对 test 文件夹下的语音进行测试！



    test()
    
    
    """
    list、dict、字符串
    
    
    
    """
