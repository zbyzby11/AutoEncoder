import torch as t
from torch import optim
from torch.nn import MSELoss
from src.model import model
from src.data_processing import data_create
from visdom import Visdom


class AE(object):
    def __init__(self,
                 batch_size=500,
                 lr=0.001,
                 training_times=500
                 ):
        """
        auto-encoder的实现
        :param batch_size:每批图片的数量
        :param lr: 学习率
        :param training_times:epoch，训练次数
        """
        self.batch_size = batch_size
        self.lr = lr
        self.training_times = training_times
        self.device = t.device('cuda' if t.cuda.is_available() else 'cpu')
        self.train_data, self.test_data = data_create(self.batch_size)
        self.model = model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss = MSELoss()

    def train(self):
        viz = Visdom()
        for epoch in range(self.training_times):
            flag = True
            # 迭代train_data
            for index, (i, _) in enumerate(self.train_data):
                i = i.to(self.device)
                x = self.model(i)
                if index % 5 == 0:
                    viz.images(
                        i[:64],
                        nrow=8,
                        win='x_train',
                        opts={'title': 'x_train'}
                    )
                    viz.images(
                        x[:64],
                        nrow=8,
                        win='x_trian_output',
                        opts={'title': 'x_trian_output'}
                    )
                loss = self.loss(x, i)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if flag:
                    print('epoch:{} || loss is:{}'.format(str(epoch), str(loss.item())))
                    flag = False
            if epoch % 5 == 0:
                # x_test = [test_batch_size,1 ,28, 28]
                # 验证test_data
                x_test, _ = next(iter(self.test_data))
                x_test = x_test.to(self.device)
                # print(x_test.size())
                # print(x_test)
                x_test_output = self.model(x_test)
                viz.images(
                    x_test[:64],
                    nrow=8,
                    win='x_test',
                    opts={'title': 'x_test'}
                )
                viz.images(
                    x_test_output[:64],
                    nrow=8,
                    win='x_test_output',
                    opts={'title': 'x_test_output'}
                )


if __name__ == '__main__':
    ae = AE()
    ae.train()
