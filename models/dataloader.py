from .utils import transform_to_tensor
import torch
import torchvision
import torchvision.transforms as transforms


__all__ = ['dataloader']

def dataloader(args, size=32):

    transform_train = transform_to_tensor(size, mode='train')
    transform_test = transform_to_tensor(size, mode='test')

    if args.dataset == 'MNIST':
        print('==> Preparing MNIST data..')
        trainset = torchvision.datasets.MNIST(root='../data/MNIST', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=16)
        print(len(trainset))

        testset = torchvision.datasets.MNIST(root='../data/MNIST', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=16)
        print(len(testset))
    elif args.dataset == 'FMNIST':
        print('==> Preparing Fashion MNIST data..')
        trainset = torchvision.datasets.FashionMNIST(root='../data/MNIST', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=16)
        print(len(trainset))

        testset = torchvision.datasets.FashionMNIST(root='../data/MNIST', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=16)
        print(len(testset))

    return trainset, testset, trainloader, testloader

