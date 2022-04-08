import torch
from torch import nn
import torch.nn.functional as F

class MyNet(torch.nn.Module): #定义了一个类
    def __init__(self): #初始化
        super(MyNet, self).__init__() #继承
        self.c1 = torch.nn.Conv2d(in_channels=3, out_channels=48, kernel_size=11, stride=4, padding=2)
        self.relu = torch.nn.ReLU() #激活函数
        self.c2 = torch.nn.Conv2d(in_channels=48, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.s2 = torch.nn.MaxPool2d(2) #池化层
        self.c3 = torch.nn.Conv2d(in_channels=128, out_channels=192, kernel_size=3, stride=1, padding=1)
        self.s3 = torch.nn.MaxPool2d(2)
        self.c4 = torch.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1)
        self.c5 = torch.nn.Conv2d(in_channels=192, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.s5 = torch.nn.MaxPool2d(kernel_size=3, stride=2)
        self.flatten = torch.nn.Flatten()
        self.f6 = torch.nn.Linear(4608, 2048)
        self.f7 = torch.nn.Linear(2048, 2048)
        self.f8 = torch.nn.Linear(2048, 1000)
        self.f9 = torch.nn.Linear(1000, 2)

    def forward(self, x):
           x = self.relu(self.c1(x))
           x = self.relu(self.c2(x))
           x = self.s2(x)
           x = self.relu(self.c3(x))
           x = self.s3(x)
           x = self.relu(self.c4(x))
           x = self.relu(self.c5(x))
           x = self.s5(x)
           x = self.flatten(x)
           x = self.f6(x)
           x = F.dropout(x, p=0.5)#防止过拟合
           x = self.f7(x)
           x = F.dropout(x, p=0.5)
           x = self.f8(x)
           x = F.dropout(x, p=0.5)

           x = self.f9(x)
           return x

if __name__ == '__main__':
    x = torch.rand([1, 3, 224, 224])#随机生成张量形式的数组
model = MyNet()#模型实体化
print(model)




