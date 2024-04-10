# -*- coding:utf-8 -*-
# @FileName : tmp.py
# @Time : 2024/3/12 18:43
# @Author : fiv
import numpy as np

# 创建两个数组
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# 使用hstack进行水平堆叠
hstack_result = np.hstack((a, b))
print("hstack result: ", hstack_result)

# 使用concatenate进行堆叠，等价于hstack
concatenate_result = np.concatenate((a, b), axis=0)
print("concatenate result: ", concatenate_result)

# 创建两个二维数组
c = np.array([[1, 2, 3], [4, 5, 6]])
d = np.array([[7, 8, 9], [10, 11, 12]])

# 使用hstack进行水平堆叠
hstack_result_2d = np.hstack((c, d))
print("hstack result for 2D arrays: \n", hstack_result_2d)

# 使用concatenate进行堆叠，等价于hstack
concatenate_result_2d = np.concatenate((c, d), axis=1)
print("concatenate result for 2D arrays: \n", concatenate_result_2d)
