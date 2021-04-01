import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import Net

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.25, 0.25, 0.25), (0.25, 0.25, 0.25))])

trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                         download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                        download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)


def learn(nt):
    optimizer = optim.SGD(nt.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(12):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = nt(inputs)
            loss = Net.criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0


net1 = Net.Net()
net2 = Net.Net()
print("The first NN:")
learn(net1)
print("\n\n")
print("The second NN:")
learn(net2)


class Way:
    w1 = 0
    w2 = 0
    theta = 0

    def __init__(self, w1, w2, theta):
        self.w1 = w1
        self.w2 = w2

    def count(self, t):
        return (1-t)**2 * self.w1 + 2*t*(1-t)*self.theta + t**2 * self.w2


def count_way(way):

    for t in range(10):
        net = Net.Net()

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data


print('Finished Training')

