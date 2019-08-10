import torch as t
from torch import nn


class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        前向传播
        :param x:输入x矩阵
        :return: 输出的矩阵
        """
        # x = [batch_size , 1, 28, 28], 一批MNIST图片
        batch_size = x.size(0)
        # 将图片转换成网络的输入,
        # x = [batch_size , 1, 28, 28] -> [batch_size, 784]
        x = x.view(batch_size, 784)
        # x = [batch_size, 784] -> [batch_size, 10]
        x = self.encoder(x)
        # x = [batch_size, 10] -> [batch_size, 784]
        x = self.decoder(x)
        # 将x重新建立得到MNSIT图片的尺寸
        x = x.view(batch_size, 1, 28, 28)
        return x


def main():
    x = t.randn((500, 1, 28, 28))
    m = model()
    output = m.forward(x)
    print(output)


if __name__ == '__main__':
    main()
