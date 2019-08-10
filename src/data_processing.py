import torch as t
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST


def data_create(batch_size):
    mnist_train = MNIST(root='../MNIST',
                        train=True,
                        download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor()
                        ]))
    mnist_test = MNIST(root='../MNIST',
                       train=False,
                       download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))
    # print(mnist_test)
    # print(mnist)
    train_data = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    test_data = DataLoader(mnist_test, batch_size=batch_size, shuffle=True)
    # print(test_data)
    return train_data, test_data


def main():
    data_create(8)


if __name__ == '__main__':
    main()
