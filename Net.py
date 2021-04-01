import torch.nn as nn
import torch.nn.functional as F


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

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.c2 * (((32 - self.kerSz + 1) // self.pl - self.kerSz + 1) // self.pl) ** 2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


criterion = nn.CrossEntropyLoss()
