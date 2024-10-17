# 1导入库
import os
import matplotlib.pyplot as plt
import pandas as pd
import torch
from tqdm import tqdm
from src.models.Encoder import VGG16_Encoder,Homo_frac_pre
from src.data.data_pre import cifar_data_load,cifar_test_data_load
import torch.nn as nn
import numpy as np
from src.models.Test import Test


def encoder_entropy(seq_probs,strength=0.01):
    ent_by_position = -torch.sum(
        seq_probs * torch.log(seq_probs + 1e-10),
        dim = 2
    )
    mean_ent_by_sequence = torch.mean(
        ent_by_position,
        dim = 1
    )
    mean_ent_by_batch = torch.mean(
        mean_ent_by_sequence,
        dim = 0
    )
    return strength * mean_ent_by_batch

def dna_seq_loss(seq_matrix,tar_gc_con=0.5):
    length = seq_matrix.shape[1]
    batch = seq_matrix.shape[0]
    gc_count = torch.sum(seq_matrix[:,:,1:3],dim=1)  # C 和 G 的概率和
    gc_count = torch.sum(gc_count, dim=-1)/length
    tar_gc_con_ = torch.full((batch,),tar_gc_con,dtype=torch.float32).to('cuda')
    gc_loss = nn.MSELoss()(gc_count,tar_gc_con_)
    return gc_loss

def set_parameter_requires_grad(model):
    for param in model.parameters():
        param.requires_grad = False


def homo_loss(seq_matrix,tar_homo_con=0.0):
    pre = homo_model(seq_matrix)
    batch = seq_matrix.shape[0]
    tar_homo = torch.full((batch,), tar_homo_con, dtype=torch.float32).to('cuda')
    loss = nn.BCELoss()(pre,tar_homo)
    return loss







class Trainer():
    # 初始化训练参数 关键参数有 待训练的模型 数据加载模块 损失函数模块 优化器模块
    def __init__(self,
                 model,
                 data_loader,
                 loss_fun,
                 optim,
                 save_dir,
                 ):
        # 初始化保存路径
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.history = os.path.join(save_dir,'history.pth')
        self.best_model = os.path.join(save_dir,'best_model.pth')
        self.loss_jpg = os.path.join(save_dir,'loss.jpg')
        # 初始化模型以及相关信息
        self.model = model
        self.metric = pd.DataFrame(columns=['train_total_loss','val_total_loss'])
        self.his_epoch = 0
        self.best_loss = np.inf

        if os.path.exists(self.history):
            # 读取历史信息
            history = torch.load(self.history)
            # 模型加载参数
            self.model.load_state_dict(history['model_last'])
            # 加载历史训练记录
            self.metric = history['metric']
            self.best_loss = history['best_loss']
            self.his_epoch = history['his_epoch']
        # 初始化数据加载模块
        self.data_loader = data_loader
        # 初始化损失函数
        self.loss_fun = loss_fun
        # 初始化优化器
        self.optim = optim(self.model.parameters(),lr=1e-4,weight_decay=1e-3)
        # 初始化提前停止模块


    def train(self,train_epoch,device='cuda'):

        self.model = self.model.to(device)
        # 开始训练
        for epoch in range(1,train_epoch+1):
            # 初始化损失和
            train_loss_sum = 0
            train_ee_loss_sum = 0
            train_gc_loss_sum = 0
            train_homo_loss_sum = 0
            train_total_loss_sum = 0
            train_step_sum = 0
            # 训练循环进度条
            train_loop = tqdm(self.data_loader[0], desc=f'Encoder Train Epoch [{epoch+self.his_epoch}/{train_epoch+self.his_epoch}]', ncols=200)
            for train_batch_data,train_batch_label in train_loop:
                # 数据提交到device
                train_batch_data = train_batch_data.to(device)
                train_batch_label = train_batch_label.to(device)
                self.optim .zero_grad()
                seq_probs,pred = self.model(train_batch_data)
                # + encoder_entropy(seq_probs)
                train_loss = self.loss_fun(pred,train_batch_label)
                train_ee_loss = encoder_entropy(seq_probs)
                # 额外损失
                # train_gc_loss = dna_seq_loss(seq_probs)
                # train_homo_loss = homo_loss(seq_probs)
                # + train_gc_loss + train_homo_loss

                train_total_loss = train_loss + train_ee_loss
                train_total_loss.backward()
                self.optim .step()

                train_step_sum += 1
                train_loss_sum += train_loss.item()
                train_ee_loss_sum += train_ee_loss.item()
                # train_gc_loss_sum += train_gc_loss.item()
                # train_homo_loss_sum += train_homo_loss.item()
                train_total_loss_sum += train_total_loss.item()
                # 'train_gc_loss': f'{train_gc_loss.item():.2f}|{train_gc_loss_sum / train_step_sum:.2f}',
                # 'train_homo_loss': f'{train_homo_loss.item():.2f}|{train_homo_loss_sum / train_step_sum:.2f}',
                train_loop.set_postfix({'train_loss': f'{train_loss.item():.2f}|{train_loss_sum / train_step_sum:.2f}',
                                        'train_ee_loss': f'{train_ee_loss.item():.2f}|{train_ee_loss_sum / train_step_sum:.2f}',

                                        'train_total_loss': f'{train_total_loss.item():.2f}|{train_total_loss_sum / train_step_sum:.2f}',
                                        }
                                        )
            train_loop.close()

            # 验证循环进度条
            self.model.eval()
            val_loss_sum = 0
            val_ee_loss_sum = 0
            val_gc_loss_sum = 0
            val_homo_loss_sum = 0
            val_total_loss_sum = 0
            val_step_sum = 0
            val_encoder_loop = tqdm(self.data_loader[1], desc=f'Encoder Val Epoch [{epoch+self.his_epoch}/{train_epoch+self.his_epoch}]', ncols=200)
            for val_batch_data, val_label in val_encoder_loop:
                with torch.no_grad():
                    val_batch_data = val_batch_data.to(device)
                    val_label = val_label.to(device)
                    seq_probs,pred = self.model(val_batch_data)
                    val_loss = self.loss_fun(pred, val_label)
                    val_ee_loss = encoder_entropy(seq_probs)
                    # val_gc_loss = dna_seq_loss(seq_probs)
                    # val_homo_loss = homo_loss(seq_probs)
                    #+ val_gc_loss + val_homo_loss
                    val_total_loss = val_loss + val_ee_loss
                val_step_sum += 1
                val_loss_sum += val_loss.item()
                val_ee_loss_sum += val_ee_loss.item()
                # val_gc_loss_sum += val_gc_loss.item()
                # val_homo_loss_sum += val_homo_loss.item()
                # 'val_gc_loss': f'{val_gc_loss.item():.2f}|{val_gc_loss_sum / val_step_sum:.2f}',
                # 'val_homo_loss': f'{val_homo_loss.item():.2f}|{val_homo_loss_sum / val_step_sum:.2f}',
                val_total_loss_sum += val_total_loss.item()

                val_encoder_loop.set_postfix({'val_loss': f'{val_loss.item():.2f}|{val_loss_sum / val_step_sum:.2f}',
                                        'ee_loss': f'{val_ee_loss.item():.2f}|{val_ee_loss_sum / val_step_sum:.2f}',

                                        'total_loss': f'{val_total_loss.item():.2f}|{val_total_loss_sum / val_step_sum:.2f}',
                                        }
                                       )
            val_encoder_loop.close()

            # 每个训练epoch保存模型以及记录损失
            new_loss = {'train_total_loss':train_total_loss_sum / train_step_sum,
                        'val_total_loss':val_total_loss_sum / val_step_sum}
            self.metric.loc[len(self.metric)] = new_loss

            if val_loss_sum / val_step_sum < self.best_loss:
                torch.save(self.model.state_dict(),self.best_model)
                self.best_loss = val_loss_sum / val_step_sum
            train_state_dict = {
                'model_last': self.model.state_dict(),
                'metric': self.metric,
                'his_epoch': self.his_epoch + epoch,
                'best_loss': self.best_loss
            }
            torch.save(train_state_dict, self.history)
            #训练结束后作图
            x_en = range(1,len(self.metric)+1)
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.title('Encoder_loss')
            plt.plot(x_en, self.metric["train_total_loss"], color='red', label="train_loss")
            plt.plot(x_en, self.metric["val_total_loss"], color='blue', label="val_loss")
            plt.legend()
            plt.savefig(self.loss_jpg)
            plt.close()

if __name__ == '__main__':
    # 实例化encoder模型
    encoder = VGG16_Encoder(80,10)
    # 实例化数据加载
    data_dir = '/home/cao/桌面/非配对检索/data/raw'
    batch_size = 200
    data_loader = cifar_data_load(data_dir,batch_size)
    # 损失函数
    loss_fun = nn.CrossEntropyLoss()
    # 优化器
    optim = torch.optim.AdamW
    # 保存路径
    save_dir = '/home/cao/桌面/非配对检索/train_model/acc_model_2'

    homo_model = Homo_frac_pre(80)
    homo_model = homo_model.to('cuda')
    best_model = '/home/cao/桌面/非配对检索/homo_pre_model/homo_model_5/best_model.pth'
    homo_model.load_state_dict(torch.load(best_model))
    set_parameter_requires_grad(homo_model)

    trainer = Trainer(
        encoder,
        data_loader,
        loss_fun,
        optim,
        save_dir
    )
    trainer.train(100)

    encoder = VGG16_Encoder(80, 10)
    data_dir = '/home/cao/桌面/非配对检索/data/raw'
    data_set, data_loader = cifar_test_data_load(data_dir, 100)
    model_path = '/home/cao/桌面/非配对检索/train_model/acc_model_2/best_model.pth'
    save_dir = '/home/cao/桌面/非配对检索/train_model/acc_model_2'
    test = Test(
        encoder,
        data_set,
        data_loader,
        model_path,
        save_dir,
        step=90,
    )

