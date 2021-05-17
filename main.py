import torch
import torchvision
import torchvision.transforms as transforms
from models import net
from config import *

import way


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


path = 'learnedNets/'
net1 = torch.load(path + 'net1.pt')
net2 = torch.load(path + 'net2.pt')

# //// to learn new nets ////

# print("The first NN:")
# net.learn(net1, trainloader)
# torch.save(net1, path + 'net1.pt')
# print("\n\n")
# print("The second NN:")
# net.learn(net2, trainloader)
# torch.save(net2, path + 'net2.pt')

# ////////////////////////////

thetaNum = 5
for theta in range(2*thetaNum + 1):
    theta -= thetaNum
    currentWay = [way.Way(net1.conv1.weight.data, net2.conv1.weight.data, theta),
                  way.Way(net1.conv1.bias.data, net2.conv1.bias.data, theta),
                  way.Way(net1.conv2.weight.data, net2.conv2.weight.data, theta),
                  way.Way(net1.conv2.bias.data, net2.conv2.bias.data, theta),
                  way.Way(net1.fc1.weight.data, net2.fc1.weight.data, theta),
                  way.Way(net1.fc1.bias.data, net2.fc1.bias.data, theta),
                  way.Way(net1.fc2.weight.data, net2.fc2.weight.data, theta),
                  way.Way(net1.fc2.bias.data, net2.fc2.bias.data, theta)]
    c = way.count_way(currentWay, trainloader)
    print(c)
    minimum = min(minimum, c)
    print(theta, ': ', minimum)

print('minimal :  ', minimum)
