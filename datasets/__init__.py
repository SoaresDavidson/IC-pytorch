from . import mnist, fashionMnist, cifar10, imageNet, cifar100

DATASET_GETTER = {
    'mnist': mnist.get_loaders,
    'fashionmnist': fashionMnist.get_loaders,
    'cifar10': cifar10.get_loaders,
    'cifar100': cifar100.get_loaders,
    'imagenet': imageNet.get_loaders
}

def load_dataset(name, batch_size):
    return DATASET_GETTER[name](batch_size)