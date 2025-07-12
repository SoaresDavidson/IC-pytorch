from . import Lenet5
from . import Lenet5BW
from . import Lenet5XNOR


MODEL_GETTER = {
    'lenet5': Lenet5.LeNet5,
    'lenet5BW': Lenet5BW.LeNet5Binary,
    'lenet5XNOR': Lenet5XNOR.LeNet5XNOR
}

def get_model(name):
    return MODEL_GETTER[name]