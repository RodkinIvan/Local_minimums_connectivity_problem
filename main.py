import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

n = torch.randn(20, 30, 10, 10)
m = nn.Conv2d(30, 40, (5, 5))
p = nn.MaxPool2d(2, 2)
print(m(n).size())


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                         download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                        download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)


class Net(nn.Module):
    c1 = 6
    c2 = 18
    kerSz = 5
    ker = (kerSz, kerSz)
    pl = 2
    l1 = 120

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


net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(4):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
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
