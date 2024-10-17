import os.path
import random

from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import dataset,DataLoader
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.models.Encoder import Homo_frac_pre


def randomly_generate_sequence(save_path,num=100000,length=80,):
    if os.path.exists(save_path):
        return pd.read_csv(save_path)
    else:
        base_lst = ['A','G','C','T']
        seq_lst = []
        frac_lst = []
        lv_count_lst = [0 for i in range(2)]
        while True:
            void_lst = []
            for i in range(length):
                id = random.randint(0,3)
                base = base_lst[id]
                void_lst.append(base)
            seq = ''.join(void_lst)
            homo_frac = calculate_homopolymer_fraction(seq)
            if homo_frac == 0 and lv_count_lst[0] <= num:
                lv_count_lst[0] += 1
                seq_lst.append(seq)
                frac_lst.append(0)
            elif homo_frac != 0 and lv_count_lst[1] <= num:
                lv_count_lst[1] += 1
                seq_lst.append(seq)
                frac_lst.append(1)
            if sum(lv_count_lst) > len(lv_count_lst)*num:
                break
        df = pd.DataFrame({'seq':seq_lst,'homo_frac':frac_lst})
        df.to_csv(save_path)
        return df
def find_intervals(values):
    return np.floor(values / 0.06).astype(int)
def calculate_homopolymer_fraction(dna_sequence):
    count = 0
    total_length = len(dna_sequence)
    i = 0

    while i < total_length:
        current_char = dna_sequence[i]
        streak_length = 1

        while i + 1 < total_length and dna_sequence[i + 1] == current_char:
            streak_length += 1
            i += 1

        if streak_length >= 4:
            count += streak_length  # 计数符合条件的均聚物

        i += 1

    # 计算均聚物分数
    homopolymer_fraction = count / total_length if total_length > 0 else 0
    return homopolymer_fraction
def seqs_to_onehots(seqs):
    seq_array = np.array(list(map(list, seqs)))
    bases = np.array(list("AGCT"))
    return np.array([(seq_array == b).T for b in bases]).T.astype(float)
def train(train_epoch,save_dir,homo_data,best_model,batch_size=320,device='cuda'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    #实例化模型
    homo_pre = Homo_frac_pre(80)
    if os.path.exists(best_model):
        homo_pre.load_state_dict(torch.load(best_model))
    homo_pre = homo_pre.to(device)
    #加载数据
    homo_train_data = randomly_generate_sequence(homo_data)
    homo_train_data = homo_train_data.sample(frac=1).reset_index(drop=True)
    seq = homo_train_data['seq'].values
    homo_frac = homo_train_data['homo_frac'].values
    homo_frac = torch.tensor(homo_frac, dtype=torch.float32)
    seq_one_hot = torch.tensor(seqs_to_onehots(seq),dtype=torch.float32)

    num_of_seq = len(homo_frac)
    num_of_train = int(num_of_seq * 0.8)

    pre_train_dataset = dataset.TensorDataset(seq_one_hot[:num_of_train], homo_frac[:num_of_train])
    pre_train_dataloader = DataLoader(pre_train_dataset, batch_size=batch_size, shuffle=True)
    pre_val_dataset = dataset.TensorDataset(seq_one_hot[num_of_train:], homo_frac[num_of_train:])
    pre_val_dataloader = DataLoader(pre_val_dataset, batch_size=batch_size, shuffle=False)

    loss_pd = pd.DataFrame(columns=['train_loss', 'val_loss'])
    optimizer = torch.optim.RMSprop(homo_pre.parameters(), 1e-3)
    loss_fun = nn.BCELoss()
    best_loss = np.inf

    best_model = os.path.join(save_dir,'best_model.pth')
    loss_jpg = os.path.join(save_dir,'loss.jpg')
    for epoch in range(1, train_epoch + 1):
        homo_pre.train()
        train_loss_sum = 0
        train_step = 0
        train_loop = tqdm(pre_train_dataloader, desc=f'Train epoch [{epoch}/{train_epoch}]', total=len(pre_train_dataloader),
                          ncols=150)
        for batch_train_data, batch_label in train_loop:
            batch_train_data = batch_train_data.to(device)
            batch_label = batch_label.to(device)
            optimizer.zero_grad()
            pre = homo_pre(batch_train_data)
            loss = loss_fun(pre, batch_label)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
            train_step += 1
            train_loop.set_postfix({'real_time_loss': loss.item(), 'average_loss': float(train_loss_sum / train_step)})
        train_loop.close()

        homo_pre.eval()
        val_loss_sum = 0
        val_step = 0
        val_loop = tqdm(pre_val_dataloader, desc=f'Val epoch [{epoch}/{train_epoch}]', total=len(pre_val_dataloader),
                        ncols=150)
        for batch_val_data, val_label in val_loop:
            val_step += 1
            with torch.no_grad():
                batch_val_data = batch_val_data.to(device)
                val_label = val_label.to(device)
                val_pre = homo_pre(batch_val_data)
                val_loss = loss_fun(val_pre, val_label)
                val_loss_sum += val_loss.item()
                val_loop.set_postfix(
                    {'real_time_loss': val_loss.item(), 'average_loss': float(val_loss_sum / val_step)})
        val_loop.close()

        new_loss = {'train_loss': train_loss_sum / train_step,
                    'val_loss': val_loss_sum / val_step}
        loss_pd.loc[len(loss_pd)] = new_loss

        if val_loss_sum / val_step < best_loss:
            torch.save(homo_pre.state_dict(), best_model)
            best_loss = val_loss_sum / val_step

        x_en = range(1, len(loss_pd) + 1)
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title('Encoder_loss')
        plt.plot(x_en, loss_pd["train_loss"], color='red', label="train_loss")
        plt.plot(x_en, loss_pd["val_loss"], color='blue', label="val_loss")
        plt.legend()
        plt.savefig(loss_jpg)
        plt.close()
def test(best_model,homo_data,device='cuda',batch_size=200):
    homo_pre = Homo_frac_pre(80)
    homo_pre = homo_pre.to(device)
    homo_pre.load_state_dict(torch.load(best_model))
    homo_train_data = randomly_generate_sequence(homo_data,num=30)
    seq = homo_train_data['seq'].values
    homo_frac = homo_train_data['homo_frac'].values
    homo_frac_tensor = torch.tensor(homo_frac, dtype=torch.float32)
    seq_one_hot = torch.tensor(seqs_to_onehots(seq), dtype=torch.float32)


    pre_train_dataset = dataset.TensorDataset(seq_one_hot, homo_frac_tensor)
    pre_train_dataloader = DataLoader(pre_train_dataset, batch_size=batch_size, shuffle=False)
    homo_pre.eval()
    pre_homo_frac = []
    val_loop = tqdm(pre_train_dataloader, desc=f'Test', total=len(pre_train_dataloader),
                    ncols=150)
    for batch_val_data, val_label in val_loop:
        with torch.no_grad():
            batch_val_data = batch_val_data.to(device)
            val_pre = homo_pre(batch_val_data).cpu().numpy().tolist()
            pre_homo_frac.extend(val_pre)
    val_loop.close()
    pre_homo_frac = np.array(pre_homo_frac).reshape(-1)

    plt.scatter(range(len(homo_frac)),homo_frac,c=np.abs(pre_homo_frac - homo_frac))
    plt.colorbar(label="|Simulated - Predicted|")
    plt.show()


if __name__ == '__main__':
    # randomly_generate_sequence('/home/cao/桌面/非配对检索/data/raw/homo_train_data/homo_train_data.csv')
    # save_dir = '/home/cao/桌面/非配对检索/homo_pre_model/homo_model_5'
    # homo_train_data = '/home/cao/桌面/非配对检索/data/raw/homo_train_data/homo_train_data.csv'
    best_model = '/home/cao/桌面/非配对检索/homo_pre_model/homo_model_5/best_model.pth'
    # train(
    #     50,
    #     save_dir,
    #     homo_train_data,
    #     best_model
    # )

    homo_data = '/home/cao/桌面/非配对检索/data/raw/homo_train_data/homo_test_data.csv'
    test(best_model,
         homo_data)