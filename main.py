# -*- coding:utf-8 -*-
# @FileName : main.py
# @Time : 2024/3/12 19:07
# @Author : fiv

from script.process import record
from env import OUTPUT_PATH

record(str(OUTPUT_PATH / 'voice.wav'), 5)
