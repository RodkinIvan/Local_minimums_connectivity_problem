import Net


class Way:
    w1 = 0
    w2 = 0
    theta = 0

    def __init__(self, w1, w2, theta):
        self.w1 = w1
        self.w2 = w2

    def count(self, t):
        return (1 - t) ** 2 * self.w1 + 2 * t * (1 - t) * self.theta + t ** 2 * self.w2


def count_way(way, train_loader, freq=10):
    net = Net.Net()
    loss = 0
    total = 0
    for t in range(freq):
        net.conv1.weight.data = way[0].count(t/freq)
        net.conv1.bias.data = way[1].count(t/freq)
        net.conv2.weight.data = way[2].count(t/freq)
        net.conv2.bias.data = way[3].count(t/freq)
        net.fc1.weight.data = way[4].count(t/freq)
        net.fc1.bias.data = way[5].count(t/freq)
        net.fc2.weight.data = way[6].count(t/freq)
        net.fc2.bias.data = way[7].count(t/freq)
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            outputs = net(inputs)
            criterion = Net.criterion(outputs, labels)
            loss += criterion.item()
        loss /= 12000
        total += loss/freq
    return total
