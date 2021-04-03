import Net


class Way:
    w1 = 0
    w2 = 0
    theta = 0

    def __init__(self, w1, w2, theta):
        self.w1 = w1
        self.w2 = w2

    def count(self, t):
        return (1-t)**2 * self.w1 + 2*t*(1-t)*self.theta + t**2 * self.w2


def count_way(way, train_loader):

    for t in range(10):
        net = Net.Net()

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data

