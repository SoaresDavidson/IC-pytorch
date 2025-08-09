import torch
import torchvision
import torchvision.transforms as t

def get_loaders(batch_size):
    train_dataset = torchvision.datasets.CIFAR10(root = './data',
                                               train = True,
                                               transform = t.Compose([
                                                      t.RandomCrop((32,32), padding=4),
                                                      t.RandomHorizontalFlip(),
                                                      t.ColorJitter(0.2, 0.2, 0.2, 0.1),
                                                      t.ToTensor(),
                                                      t.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
                                                      ]),
                                               download = True)
    
    
    test_dataset = torchvision.datasets.CIFAR10(root = './data',
                                                train = False,
                                                transform = t.Compose([
                                                        # t.Resize((224, 224)),
                                                        t.ToTensor(),
                                                        t.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
                                                        ]),
                                                download=True)
        
        
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                                batch_size = batch_size,
                                                shuffle = True,
                                                pin_memory=True,
                                                num_workers=16
                                                )
        
        
    test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                                batch_size = batch_size,
                                                shuffle = False,
                                                pin_memory=True,
                                                num_workers=16
                                                )
    
    return train_loader, test_loader