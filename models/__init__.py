from . import Lenet5
from . import Lenet5Binarized
from . import Lenet5Moderna
from . import Lenet5ModernaBinarizada

MODEL_GETTER = {
    'lenet5': Lenet5.LeNet5,
    'lenet5B': Lenet5Binarized.LeNet5Binary,
    'lenet5M': Lenet5Moderna.LeNet5,
    'lenet5MB': Lenet5ModernaBinarizada.LeNet5Binary
}

def get_model(name):
    return MODEL_GETTER[name]