import random

import torch
from src.data.data_pre import cifar_test_data_load
import numpy as np
from torchvision import datasets, transforms
from PIL import Image
import torchvision
import numpy as np
import pandas as pd

from RNA import *

# 输入DNA序列sequence = "gcatttgtttggttaaccaagcatgcagcgttcctaaaacgaccatgctttaccccccgctttagccctgaggaaacccacacatttcgtacttcttctttctaaaaccctgactgctgcgactagacttaagcggagtaacgcacgtcagcaattgatgggttgtcaagcaccatgcgatctatttgtcgacgccccgg"



sequence = "gcatttgtttggttaaccaagcatgcagcgttcctaaaacgaccatgctttaccccccgctttagccctgaggaaacccacacatttcgtacttcttctttctaaaaccctgactgctgcgactagacttaagcggagtaacgcacgtcagcaattgatgggttgtcaagcaccatgcgatctatttgtcgacgccccgg"



RNA.cvar.temperature = 59.1

# 创建一个folding parameters对象
# 默认情况下，RNAfold会使用标准参数，我们可以通过设置以下参数来禁用GU和打印


# 进行folding
structure, energy = RNA.fold(sequence)

print("预测的二级结构:", structure)
print("自由能 (kcal/mol):", energy)











