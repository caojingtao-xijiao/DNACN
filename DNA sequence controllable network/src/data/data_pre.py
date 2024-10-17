import os
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset,Subset
from sklearn.model_selection import train_test_split
# 原始图像预处理，标签预处理后 输出

#自定义dataset,数据预处理
class Image_Dataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_filenames = os.listdir(image_dir)  # 获取目录中的所有文件名
    def __len__(self):
        return len(self.image_filenames)
    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_filenames[idx])
        image = datasets.folder.default_loader(img_name)  # 加载图像
        label = self._get_label_from_filename(self.image_filenames[idx])  # 从文件名获取标签
        image = self.transform(image)
        return image, label
    def _get_label_from_filename(self, filename):
        # 这里可以根据文件名提取标签，假设标签是文件名中的某一部分
        return int(filename.split('_')[0])  # 示例：文件名为 "0_image.jpg" 返回标签 0
    def transform(self,image):
        trans = transforms.Compose([
            transforms.Resize((128, 128)),  # 调整图像大小
            transforms.ToTensor(),  # 转换为 Tensor
            transforms.Normalize((0.5,), (0.5,)),  # 标准化
        ])
        return trans(image)


def cifar_data_load(data_dir,batch_size,):
    #图像预处理转换
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomCrop(32, padding=4),  # 随机裁剪，带边缘填充
        transforms.Resize((224, 224)),  # 调整大小到 224x224
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化
    ])
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整大小到 224x224
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化
    ])
    #加载训练数据
    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True,
                                            transform=transform_train)
    #提取数据与标签
    train_data, train_labels = trainset.data, trainset.targets
    #随机划分训练与验证
    train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=0.2,shuffle=True,
                                                                      random_state=42)
    train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=False,
                                                 transform=transform_train)
    train_dataset.data, train_dataset.targets = train_data, train_labels
    val_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=False,
                                               transform=transform_val)
    val_dataset.data, val_dataset.targets = val_data, val_labels
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader,val_loader

def cifar_test_data_load(data_dir,batch_size,rate=0.3):
    # 图像预处理转换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整大小到 224x224
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化
    ])
    test_data = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True,
                                            transform=transform)
    # total_samples = len(test_data)
    # num_samples = int(total_samples * rate)

    # # 创建一个新的子集，只包含前 30% 的数据
    # subset_indices = list(range(num_samples))  # 前 num_samples 的索引
    # remain_indices = list(range(num_samples,total_samples))
    # subset = Subset(test_data, subset_indices)
    # remain_set = Subset(test_data,remain_indices)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return test_data,test_loader

