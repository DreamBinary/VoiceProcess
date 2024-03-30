import pickle
from matplotlib import pyplot as plt
from pathlib import Path


def plot_loss_bylogtxt(txt_path, ver):
    '''
    # 写一个适合 有多步训练的 loss look函数。
    :param txt_path:
    :param ver:
    :return:
    '''

    '''
    把一个A:B称为一个loss数据项
    log.txt 大概是下面这个样子 (默认有 写“当前进程的内存使用：”这个数据项)

    hparams记录....
    ****************
    step:1.0000000000,ge2eloss:230.0902709961,当前进程的内存使用：2.4297 GB
    step:2.0000000000,ge2eloss:228.5778961182,当前进程的内存使用：2.4414 GB
    step:3.0000000000,ge2eloss:218.2294921875,当前进程的内存使用：2.4419 GB
    ....
    epoch:0,step:3.0000000000,anotherloss1:50,anotherloss2:30,当前进程的内存使用：2.4419 GB
    epoch:0,step:4.0000000000,anotherloss1:55,anotherloss2:37,当前进程的内存使用：2.4419 GB
    .....


    '''
    with open(txt_path, encoding='utf-8') as f2:
        lines = list(f2.readlines())
    loss_lines = lines[:]  # 将这一行下面所有行保存下来。用于作图

    # 1.2 寻找 数据项 的key
    loss_keys = []
    for line in loss_lines:
        l_keys = [k_v.split(':')[0] for k_v in line.split(',')[:-1]]
        ## 这里之所以取 【：-1】是因为我们要忽略掉 “当前进程的内存使用”数据项
        ## 如果你没有这一项，那么可以 写成 line.split(',')
        ##  k_v.split(':')[0]  是从 A:B中取出A.

        loss_keys += l_keys
    loss_keys = list(set(loss_keys))  ## 取集合得到 字典的键。
    # 创建字典
    loss_dict = {}
    for k in loss_keys:
        loss_dict[k] = []

    # print(loss_keys)
    # 1.3 接下来，将每一个键 对应的 一组数值，加入到列表中。 比如 loss:[8,7,6,5,4 .....]
    # 这些数值还是要逐行去读取loss_lines。
    for line in loss_lines:
        l_keys_values = [(k_v.split(':')[0], k_v.split(':')[1]) for k_v in line.split(',')[:-1]]
        # print(l_keys_values) ### [('step', '1.0000000000'), ('ge2eloss', '230.0902709961')]
        # exit()
        ### 将每个 A:B,添加到字典。
        for a_b in l_keys_values:
            loss_dict[a_b[0]].append(float(a_b[1]))  # 注意字符转浮点数
    #####
    ## 则loss dict 保存了全部数据项的values和keys。
    '''
    loss dict:
    {
    'epoch':[0,0,0,0,0,...]
    'step':[0,1,2,3....]
    'loss1':[10,8,7,6,....]
    'loss2':[100,90,....]
    }

    '''
    # print(loss_dict)
    ###  作图，并保存。
    ## 创建保存图片的文件夹
    save_file_dir = Path('Lossfigure/Loss_Curves_Figures_{}'.format(ver))
    save_file_dir.mkdir(parents=True, exist_ok=True)

    ### 作图
    for k, v in loss_dict.items():
        plt.figure()
        plt.title(str(k))
        # 画图的数值截断
        # if max(v)>= 100:
        #     v = [x for x in v if x < 100]
        plt.plot(range(len(v)), v)
        plt.savefig(str(save_file_dir / f"{k}.png"))
        plt.show()



if __name__ == '__main__':
    ver = "e1"
    pa = f'e1/train_log.txt'
    plot_loss_bylogtxt(pa, ver)
    pass
