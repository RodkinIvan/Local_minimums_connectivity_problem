import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from config import *


class Net(nn.Module):
    c1 = 6
    c2 = 16
    kerSz = 5
    ker = (kerSz, kerSz)
    pl = 2
    l1 = 1200

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, self.c1, self.ker)
        self.pool = nn.MaxPool2d(self.pl, self.pl)
        self.conv2 = nn.Conv2d(self.c1, self.c2, self.ker)
        self.fc1 = nn.Linear(self.c2 * (((32 - self.kerSz + 1) // self.pl - self.kerSz + 1) // self.pl) ** 2, self.l1)
        self.fc2 = nn.Linear(self.l1, 100)
        self.conv1.weight.data = self.conv1.weight.data.to(cuda)
        self.conv1.bias.data = self.conv1.bias.data.to(cuda)
        self.conv2.weight.data = self.conv2.weight.data.to(cuda)
        self.conv2.bias.data = self.conv2.bias.data.to(cuda)
        self.fc1.weight.data = self.fc1.weight.data.to(cuda)
        self.fc1.bias.data = self.fc1.bias.data.to(cuda)
        self.fc2.weight.data = self.fc2.weight.data.to(cuda)
        self.fc2.bias.data = self.fc2.bias.data.to(cuda)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.c2 * (((32 - self.kerSz + 1) // self.pl - self.kerSz + 1) // self.pl) ** 2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


criterion = nn.CrossEntropyLoss()


def learn(nt, train_loader, epochs=12):
    optimizer = optim.SGD(nt.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            labels = labels.to(cuda)
            inputs = inputs.to(cuda)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = nt(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')
