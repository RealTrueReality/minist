import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 定义数据转换
transformer = transforms.Compose([transforms.ToTensor(),
                                  transforms.Resize(32),
                                  ])

# 下载和加载数据集
train_data = datasets.MNIST(root='./MNIST', train=True, download=True, transform=transformer)
test_data = datasets.MNIST(root='./MNIST', train=False, download=True, transform=transformer)

# 数据集大小
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练集大小为：", train_data_size)
print("测试集大小为：", test_data_size)

# 创建数据加载器
train_data_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_data_loader = DataLoader(test_data, batch_size=64, shuffle=True)

# 定义网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(6),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(120),
            nn.Flatten(),
            nn.Dropout(0.5),  # 添加Dropout层
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # 添加Dropout层
            nn.Linear(in_features=84, out_features=10),
        )

    def forward(self, x):
        x = self.model(x)
        return x

# 创建模型实例
net = Net()

# 将模型移动到GPU
net = net.to('cuda')

# 定义损失函数
losser = nn.CrossEntropyLoss()
losser=losser.to('cuda')

# 定义优化器
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)  # 设置初始学习率为0.001

# 学习率调度器
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)  # 每2个epoch将学习率乘以0.1

# 训练模型
if __name__ == '__main__':
    train_step = 0
    for epoch in range(5):
        net.train()
        for train_data in train_data_loader:
            imgs, targets = train_data
            # 将数据和目标移动到GPU
            imgs, targets = imgs.to('cuda'), targets.to('cuda')
            outputs = net(imgs)
            loss = losser(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_step += 1
            if train_step % 100 == 0:
                print('Epoch:', epoch)
                print('train_step:', train_step)
                print('Loss:', loss.item())
        accuracy = 0
        total_accuracy = 0
        net.eval()
        with torch.no_grad():
            for test_data in test_data_loader:
                imgs, targets = test_data
                # 将数据和目标移动到GPU
                imgs, targets = imgs.to('cuda'), targets.to('cuda')
                outputs = net(imgs)
                accuracy = outputs.argmax(axis=1).eq(targets).sum().item()
                total_accuracy += accuracy
            print(f"第{epoch+1}:total_accuracy:", total_accuracy/test_data_size)

        # 输出测试集中的图像
        sample_images, sample_targets = next(iter(test_data_loader))
        sample_images = sample_images.to('cuda')
        sample_outputs = net(sample_images)
        sample_predictions = sample_outputs.argmax(axis=1)
        for i in range(6):
            plt.imshow(sample_images[i].cpu().squeeze().numpy(), cmap='gray')
            plt.title(f"Prediction: {sample_predictions[i]}, True Label: {sample_targets[i]}")
            plt.show()

        # 更新学习率
        scheduler.step()