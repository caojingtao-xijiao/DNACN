# 导入模型
import os
import random
import time
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm
from src.models.Encoder import VGG16_Encoder
from src.data.data_pre import cifar_test_data_load
from src.models.Simulator import simulator
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

class Test():
    def __init__(self,
                 model,
                 data_set,
                 data_loader,
                 model_path,
                 save_dir,
                 step,
                 device='cuda',
                 ):
        # 初始化
        self.device = device
        self.model = model
        self.model.load_state_dict(torch.load(model_path))
        self.model = self.model.to(device)
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.data_set = data_set
        self.data_loader = data_loader
        self.step = step
        self.dna_lib_path = os.path.join(save_dir,'dna_lib.csv')
        self.query_lib_path = os.path.join(save_dir,'query_lib.csv')
        self.cls_confusion_path = os.path.join(save_dir,'cls_confusion')
        self.cifar10_classes = np.array(data_set.classes)
        #初始化时即构造好dna——lib
        self.DNA_library = self.construct_dna_lib()
        self.cls_confusion = self.cls_confusion_matrix()
        self.acc = self.metric_calu()

    def construct_dna_lib(self):
        if not os.path.exists(self.dna_lib_path):
            test_encoder_loop = tqdm(self.data_loader, desc=f'Construct_DNA_Lib', ncols=150)
            DNA_library = pd.DataFrame(columns=['dna_sequence'])
            self.model.eval()
            step = 0
            for test_batch_data, test_label in test_encoder_loop:
                with torch.no_grad():
                    test_batch_data = test_batch_data.to(self.device)
                    batch_dna_seq = self.model.encode(test_batch_data)
                    frame = pd.DataFrame({'dna_sequence':batch_dna_seq})
                    DNA_library = pd.concat([DNA_library,frame],ignore_index=True)
                step += 1
                if step == self.step:
                    break
            #计算GC含量与均聚物分数
            gc_content_lst = []
            homo_frac_lst = []
            for i in DNA_library['dna_sequence']:
                gc_content_lst.append(self.calculate_gc_content(i))
                homo_frac_lst.append(self.calculate_homopolymer_fraction(i))
            DNA_library['gc_content'] = gc_content_lst
            DNA_library['homo_frac'] = homo_frac_lst
            a = DNA_library['gc_content']
            b = DNA_library['homo_frac']
            print(sum(a < 40))
            print(sum(a > 60))
            print(sum(b > 0))
            print(DNA_library['homo_frac'][b > 0].mean())
            DNA_library.to_csv(self.dna_lib_path)
            return DNA_library
        else:
            DNA_library = pd.read_csv(self.dna_lib_path)
            a = DNA_library['gc_content']
            print(sum(a<40))
            print(sum(a>60))
            b = DNA_library['homo_frac']
            print(sum(b > 0))
            print(DNA_library['homo_frac'][b>0].mean())

            return DNA_library

    def calculate_gc_content(self,dna_sequence):
        dna_sequence = dna_sequence.upper()
        dna_sequence = dna_sequence.upper()
        # 计算 GC 的数量
        g_count = dna_sequence.count('G')
        c_count = dna_sequence.count('C')
        # 计算总碱基数
        total_bases = len(dna_sequence)
        # 计算 GC 含量百分比
        if total_bases == 0:
            return 0.0
        gc_content = (g_count + c_count) / total_bases * 100
        return gc_content

    def calculate_homopolymer_fraction(self,dna_sequence):
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
    #将query——img转换为图像
    def construct_query_lib(self,query_img):
        print('query图像转DNA...')
        if os.path.exists(self.query_lib_path):
            DNA_quary_lib = pd.read_csv(self.query_lib_path,index_col=0)
        else:
            DNA_quary_lib = pd.DataFrame(columns=['seq'])
        quary_id = os.path.basename(query_img)[:-4]
        self.model.eval()
        query_seq,query_fe = self.model.query_encode(query_img)
        DNA_quary_lib.loc[quary_id] = query_seq
        DNA_quary_lib.to_csv(self.query_lib_path)
        print('成功了。')
        return query_seq,query_fe
    #杂交预测
    def hybridization_search_simulation(self,query_img):
        query_seq,query_fe= self.construct_query_lib(query_img)
        quary_id = os.path.basename(query_img)[:-4]
        query_simu_save_path = os.path.join(self.save_dir,f'{quary_id}.csv')
        if os.path.exists(query_simu_save_path):
            return pd.read_csv(query_simu_save_path)
        else:
            # 计算模拟杂交分数
            print('模拟杂交分数中...')
            start_time = time.time()
            dna_library = pd.read_csv(self.dna_lib_path)
            quary = np.repeat([query_seq], len(dna_library))
            quary_dna_lib_pairs = pd.DataFrame({
                "dna_library": dna_library['dna_sequence'].values,
                "quary": quary
            })
            simu_yields = simulator(quary_dna_lib_pairs)
            print(f'模拟结束。时间消耗：{time.time()-start_time}')

            # 计算欧式距离
            simi_lst = []
            self.model.eval()
            step = 0
            cls_list = []
            for test_data,test_label in self.data_loader:
                cls_set = self.cifar10_classes[test_label].tolist()
                cls_list.extend(cls_set)
                test_data = test_data.to(self.device)
                test_fe = self.model.feature_extrat(test_data)
                for i in test_fe:
                    simi_lst.append(torch.norm(query_fe - i).item())
                step += 1
                if step == self.step:
                    break

            #合并在一个dataframe中
            result_df = pd.DataFrame({'yields':simu_yields,'simi':simi_lst,'cls':cls_list})
            result_df.to_csv(query_simu_save_path)
            return result_df
    #
    def top_image_retrieve(self,query_img,top_count=10):
        result_df = self.hybridization_search_simulation(query_img)
        quary_id = os.path.basename(query_img)[:-4]
        sorted_simu_yields = result_df.sort_values(by='yields', ascending=False)
        sorted_simu_yields.to_csv(os.path.join(self.save_dir,f'{quary_id}_sorted.csv'))
        sorted_simu_yields_index = sorted_simu_yields.index.tolist()
        top_sorted = sorted_simu_yields_index[:top_count]
        save_dir = os.path.join(self.save_dir,quary_id)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for top in top_sorted:
            image_tensor = self.data_set.data[top]
            image = Image.fromarray(image_tensor)
            label = self.data_set.targets[top]
            cifar10_classes = self.data_set.classes
            cls = cifar10_classes[label]
            image.save(os.path.join(save_dir,f'{cls}_{top}.jpg'))

        # 作散点图
        scatter_save_path = os.path.join(save_dir, f'{quary_id}_Scatter.jpg')
        x = result_df['simi']
        y = result_df['yields']
        plt.xlabel('Cosine similarity')
        plt.ylabel('Nupack Simulated yield')
        plt.title(quary_id)
        plt.scatter(x, y)
        plt.savefig(scatter_save_path)
        plt.close()
        # 作小提琴图
        violi_save_path = os.path.join(save_dir, f'{quary_id}_Violi.jpg')
        simi = result_df['simi'].values
        print(type(simi[0]))

        # x_ticks = [
        #     '[0,5]',
        #     '[5,10]',
        #     '[10,15]',
        #     '[15,20]',
        #     '[20,25]',
        # ]
        x_ticks = [str([i,i+5]) for i in range(0,100,5)]


        cos_simi_bool_ = [(i<simi)&(simi<=i+5) for i in range(0,100,5)]


        simi_level_lst = [result_df[cos_simi]['yields'].values for cos_simi in cos_simi_bool_]
        simi_level_lst_new = []

        x_ticks_new = []
        for i in range(len(simi_level_lst)):
            if len(simi_level_lst[i]) != 0:
                simi_level_lst_new.append(simi_level_lst[i])
                x_ticks_new.append(x_ticks[i])

        positions = [i for i in range(1, len(x_ticks_new) + 1)]


        plt.violinplot(simi_level_lst_new, positions=positions)
        plt.xticks(positions, x_ticks_new)
        plt.xlabel('Cosine similarity')
        plt.ylabel('Nupack Simulated yield')
        plt.title(quary_id)
        plt.savefig(violi_save_path)
        plt.close()
        return top_sorted

    def generate_images(self):
        id = random.randint(self.step*250,len(self.data_set)-1)
        image_tensor = self.data_set.data[id]
        label = self.data_set.targets[id]
        image = Image.fromarray(image_tensor)
        cifar10_classes = self.data_set.classes
        cls = cifar10_classes[label]
        image.save(os.path.join(self.save_dir, f'{cls}_{id}.jpg'))
        print(f'query图像是【{cls}_{id}.jpg】')
        return os.path.join(self.save_dir, f'{cls}_{id}.jpg')
    def cls_confusion_matrix(self):
        if os.path.exists(self.cls_confusion_path):
            return pd.read_csv(self.cls_confusion_path)
        else:
            test_encoder_loop = tqdm(self.data_loader, desc=f'Confusion Matrix', ncols=150)
            DNA_library = pd.DataFrame(columns=['pre','true'])
            self.model.eval()
            step = 0
            for test_batch_data, test_label in test_encoder_loop:
                with torch.no_grad():
                    test_batch_data = test_batch_data.to(self.device)
                    cls= self.model.cls(test_batch_data)
                    test_label = test_label.numpy()
                    frame = pd.DataFrame({'pre': cls,'true':test_label})
                    DNA_library = pd.concat([DNA_library, frame], ignore_index=True)
                step += 1
                if step == self.step:
                    break
            print(DNA_library['true'].values.tolist())
            print(DNA_library['pre'].values.tolist())
            cls_confusion_matrix = confusion_matrix(DNA_library['true'].values.tolist(),DNA_library['pre'].values.tolist())

            cls_true_label = self.data_set.classes
            cls_pre_label = [i+'pre' for i in cls_true_label]

            cls_confusion_matrix_df = pd.DataFrame(cls_confusion_matrix,columns=cls_pre_label,index=cls_true_label)
            cls_confusion_matrix_df.to_csv(self.cls_confusion_path)
            return cls_confusion_matrix_df
    def metric_calu(self):
        #计算准确率
        matrix = self.cls_confusion.values
        acc = np.sum(matrix.diagonal())/np.sum(matrix)
        print(acc)
        return acc















if __name__ == '__main__':
    # 实例化测试器
    # 实例化模型
    encoder = VGG16_Encoder(80,10)
    data_dir = '/home/cao/桌面/非配对检索/data/raw'
    data_set,data_loader = cifar_test_data_load(data_dir,100)
    model_path = '/home/cao/桌面/非配对检索/train_model/model_new_1/best_model.pth'
    save_dir = '/home/cao/桌面/非配对检索/train_model/model_new_1'
    test = Test(
        encoder,
        data_set,
        data_loader,
        model_path,
        save_dir,
        step=90,
    )
    # query_img = test.generate_images()
    # test.top_image_retrieve(query_img)











