import glob
import os
# 用于获取常见字母及字符规范化
import string
import unicodedata
import random
import time
import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plot

all_letters = string.ascii_letters + " .,;'"
# 获取常用字符数量
n_letters = len(all_letters)


# 去掉一些语言中的重音标记
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )