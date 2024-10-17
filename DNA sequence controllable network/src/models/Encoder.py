import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models
from einops.layers.torch import Rearrange
from torchvision import  transforms
from PIL import Image

#构造模型
class VGG16_Encoder(nn.Module):
    def __init__(self, dna_len, cls_num):
        super(VGG16_Encoder, self).__init__()
        self.dna_len = dna_len
        # 导入Image——net预训练的VGG16模型
        vgg16 = models.vgg16(weights='IMAGENET1K_V1')
        self.feature_ex = nn.Sequential(
            vgg16.features,
            vgg16.avgpool,
            nn.Flatten(),
            vgg16.classifier[:-1]
        )
        # 构造编码器
        self.encoder = nn.Sequential(
            nn.Linear(4096,4*dna_len,bias=True),
            Rearrange('b (h w) -> b h w ',h=dna_len,w=4),
            nn.Softmax(dim=-1),
        )
        # 构造分类器
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4*dna_len,cls_num),
        )
        self.set_parameter_requires_grad(self.feature_ex)
        self.bases = np.array(list("AGCT"))
    # 前向传播
    def forward(self, x):
        x = self.feature_ex(x)
        x = self.encoder(x)
        y = self.classifier(x)
        return x,y
    # 固定参数，不进行训练
    def set_parameter_requires_grad(self,model,bool=False):
        for param in model.parameters():
            param.requires_grad = bool
    # 使用编码器编码序列【b，3，224，224】
    def encode(self,x):
        x = self.feature_ex(x)
        x = self.encoder(x)
        x = torch.argmax(x,dim=-1).reshape(-1,self.dna_len)#x->[b,dna_len]
        x = np.array(x.cpu())
        lst = []
        for i in self.bases[x]:
            lst.append(''.join(list(i)))
        return np.array(lst)
    def cls(self,x):
        x = self.forward(x)[1]
        x = torch.argmax(x,dim=-1).reshape(-1)
        x = np.array(x.cpu())
        return x
    def query_encode(self,query_img):
        image = Image.open(query_img)
        # 定义转换操作
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 调整大小到 224x224
            transforms.ToTensor(),  # 转换为张量
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化
        ])
        tensor_image = transform(image).unsqueeze(0).to('cuda')
        query_seq, query_fe = self.encode(tensor_image),self.feature_extrat(tensor_image)
        return query_seq,query_fe
    def feature_extrat(self,x):
        return self.feature_ex(x).cpu()

class Homo_frac_pre(nn.Module):
    def __init__(self,length):
        super(Homo_frac_pre, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,4,(4,1),1),
            nn.ReLU(),
            nn.MaxPool2d((4,4),1),
            nn.Flatten(),
            nn.Linear(74*4,1),
            nn.Sigmoid(),
            nn.Flatten(0)
        )
        self.initialize_parameters()
    def forward(self,x):
        x = torch.unsqueeze(x, 1)
        return self.conv(x)

    def initialize_parameters(self):
        # 使用 Kaiming 初始化（He 初始化）对于 ReLU 激活函数
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m,nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)









# class VGG16_Encoder(nn.Module):
#     def __init__(self, dna_len, cls_num):
#         super(VGG16_Encoder, self).__init__()
#         self.dna_len = dna_len
#         # 导入Image——net预训练的VGG16模型
#         vgg16 = models.vgg16(pretrained=True)
#         self.feature_ex = nn.Sequential(
#             vgg16.features,
#             vgg16.avgpool,
#             nn.Flatten(),
#             vgg16.classifier[:-1]
#         )
#         # 构造编码器
#         self.encoder = nn.Sequential(
#             nn.Linear(4096,dna_len*2,bias=True),
#             nn.Sigmoid()
#         )
#         # 构造分类器
#         self.classifier = nn.Sequential(
#             nn.Linear(2*dna_len,cls_num,bias=True),
#         )
#         self.set_parameter_requires_grad(self.feature_ex)
#         self.base_dict = {
#             '11':'G',
#             '10':'C',
#             '01':'A',
#             '00':'T'
#         }
#
#     # 前向传播
#     def forward(self, x):
#         x = self.feature_ex(x)
#         x = self.encoder(x)
#         y = self.classifier(x)
#         return y
#     # 固定参数，不进行训练
#     def set_parameter_requires_grad(self,model):
#         for param in model.parameters():
#             param.requires_grad = False
#     # 使用编码器编码序列【b，3，224，224】
#     def encode(self,x):
#         x = self.feature_ex(x)
#         x = self.encoder(x)
#         x = (x>0.5).int()
#         x = x.reshape(-1,2,self.dna_len)
#         x = x.cpu()
#         lst_ = []
#         for i in x:
#             i = i.T
#             lst = []
#             for j in i:
#                 if j[0] == 1:
#                     if j[1] == 1:
#                         lst.append('G')
#                     else:
#                         lst.append('C')
#                 else:
#                     if j[1] == 1:
#                         lst.append('A')
#                     else:
#                         lst.append('T')
#             base_seq = ''.join(lst)
#             lst_.append(base_seq)
#         return np.array(lst_)







if __name__ == '__main__':
    #测试模型
    a = torch.rand(1,3,224,224)
    mo = VGG16_Encoder(80,10)
    print(mo.encode(a))
    # dd = torch.load('/home/cao/桌面/非配对检索/train_model/model_2/best_model.pth')
    # mo.load_state_dict(dd)
    # print(mo.encode(a))