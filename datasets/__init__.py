from . import mnist
from . import fashionMnist
from . import cifar10
DATASET_GETTER = {
    'mnist': mnist.get_loaders,
    'fashionmnist': fashionMnist.get_loaders,
    'cifar10': cifar10.get_loaders
}

def load_dataset(name, batch_size):
    return DATASET_GETTER[name](batch_size)