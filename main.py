import torch
import torchvision
import torchvision.transforms as transforms
import Net
import Way

torch.cuda.set_device(0)

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

net1 = Net.Net()
net2 = Net.Net()
print("The first NN:")
Net.learn(net1, trainloader)
print("\n\n")
print("The second NN:")
Net.learn(net2, trainloader)
minimum = 1000
for theta in range(5):
    way = [Way.Way(net1.conv1.weight.data, net2.conv1.weight.data, theta),
           Way.Way(net1.conv1.bias.data, net2.conv1.bias.data, theta),
           Way.Way(net1.conv2.weight.data, net2.conv2.weight.data, theta),
           Way.Way(net1.conv2.bias.data, net2.conv2.bias.data, theta),
           Way.Way(net1.fc1.weight.data, net2.fc1.weight.data, theta),
           Way.Way(net1.fc1.bias.data, net2.fc1.bias.data, theta),
           Way.Way(net1.fc2.weight.data, net2.fc2.weight.data, theta),
           Way.Way(net1.fc2.bias.data, net2.fc2.bias.data, theta)]
    minimum = min(minimum, Way.count_way(way, trainloader))
    print(theta, ': ', minimum)

print('minimal :  ', minimum)
