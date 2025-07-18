import torch
import torchvision
import torchvision.transforms as transforms
from util import plot_batch
batch_size = 64

train_dataset = torchvision.datasets.CIFAR100(root = './data',
                                            train = True,
                                            transform = transforms.Compose([
                                            transforms.ToTensor(),
                                        ]  )
                                        )


test_dataset = torchvision.datasets.CIFAR100(root = './data',
                                            train = False,
                                            transform = transforms.Compose([
                                                    transforms.ToTensor(),
                                                ]),
                                            )
    
    
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                            batch_size = batch_size,
                                            shuffle = True)
    
    
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                            batch_size = batch_size,
                                            shuffle = False)

plot_batch(train_loader, train_dataset.classes)
