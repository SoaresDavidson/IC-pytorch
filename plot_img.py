import torchvision.transforms as transforms
from datasets import load_dataset
from util import plot_batch
batch_size = 64

train_loader, test_loader = load_dataset(name='imagenet' ,batch_size=batch_size)

plot_batch(train_loader, 's')
