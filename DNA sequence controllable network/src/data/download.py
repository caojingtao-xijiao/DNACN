import torchvision
import torchvision.transforms as transforms
import torch
from sklearn.model_selection import train_test_split
# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
# 加载数据集
trainset = torchvision.datasets.CIFAR10(root='/home/cao/桌面/非配对检索/data/raw', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='/home/cao/桌面/非配对检索/data/raw', train=False, download=True, transform=transform)

# 将数据和标签分开
train_data, train_labels = trainset.data, trainset.targets
print(train_data.shape)
print(len(train_labels))

# 2. 随机划分数据
train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)

# 3. 创建新的数据集
train_dataset = torchvision.datasets.CIFAR10(root='/home/cao/桌面/非配对检索/data/raw', train=False, download=False, transform=transform)
train_dataset.data, train_dataset.targets = train_data, train_labels
print(train_dataset.data.shape)

val_dataset = torchvision.datasets.CIFAR10(root='/home/cao/桌面/非配对检索/data/raw', train=False, download=False, transform=transform)
val_dataset.data, val_dataset.targets = val_data, val_labels
print(val_dataset.data.shape)

# 4. 创建数据加载器
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)
#
# # 现在可以使用 train_loader 和 val_loader 进行训练和验证
#
# testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)
for i,j in train_loader:
    print(i.shape)
    print(j.shape)
