import torch
import torchvision
from torch.utils.data import DataLoader
from config import config
from utils import plot_images


def get_mnist_data(args):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((args.img_size, args.img_size)), 
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,)) 
    ])
    dataset = torchvision.datasets.MNIST(root=args.dataset_path, train=True, download=True, transform=transforms)
    return dataset


def get_custom_data(args):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((args.img_size, args.img_size)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = torchvision.datasets.ImageFolder(root=args.dataset_path, transform=transforms)
    return dataset


def get_data(args=config):
    if args.dataset_name == "MNIST":
        dataset = get_mnist_data(args)
    elif args.dataset_name == "CRYPTOPUNKS":        
        dataset = get_custom_data(args)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    return dataloader