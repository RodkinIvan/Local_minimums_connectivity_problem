import torch
from models import net
from config import *


class Way:
    w1 = 0
    w2 = 0
    theta = 0

    def __init__(self, w1, w2, theta):
        self.w1 = w1
        self.w2 = w2
        self.theta = theta

    def count(self, t):
        one = torch.ones(self.w1.size())
        one = one.to(cuda)
        return ((1 - t) ** 2 * self.w1) + (2 * t * (1 - t) * self.theta * one) + (t ** 2 * self.w2)


def count_way(way, train_loader, freq=3):
    network = net.Net()
    loss = 0
    total = 0

    for t in range(freq):
        network.conv1.weight.data = way[0].count(t / freq)
        network.conv1.bias.data = way[1].count(t / freq)
        network.conv2.weight.data = way[2].count(t / freq)
        network.conv2.bias.data = way[3].count(t / freq)
        network.fc1.weight.data = way[4].count(t / freq)
        network.fc1.bias.data = way[5].count(t / freq)
        network.fc2.weight.data = way[6].count(t / freq)
        network.fc2.bias.data = way[7].count(t / freq)
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            labels = labels.to(net.cuda)
            inputs = inputs.to(net.cuda)
            outputs = network(inputs)
            criterion = net.criterion(outputs, labels)
            loss += criterion.item()
        loss /= 12000
        total += loss / freq
    return total
