import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader
import os

def get_loaders(batch_size):
    transform_train = transforms.Compose([transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ColorJitter(brightness=0.5,
                                                                contrast=0.5,
                                                                saturation=0.3),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])])
    # transform_eval = transforms.Compose([
    #                                     transforms.Resize(256),
    #                                     transforms.CenterCrop(224),
    #                                     transforms.ToTensor(),
    #                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                                           std=[0.229, 0.224, 0.225])
    # ])
    path = os.getcwd() + "/data/1/imagenet1k"
    full_ds = torchvision.datasets.ImageFolder(path, transform=transform_train)

    train_size = int(0.8 * len(full_ds))
    eval_size = len(full_ds) - train_size

    train_ds, eval_ds = random_split(full_ds, [train_size, eval_size])

    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        dataset=eval_ds,
        batch_size=batch_size,
    )

    return train_loader, test_loader
