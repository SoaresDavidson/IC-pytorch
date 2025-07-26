from . import Lenet5, Lenet5BW, Lenet5XNOR, AlexNet, Lenet5Q



MODEL_GETTER = {
    'lenet5': Lenet5.LeNet5,
    'lenet5BW': Lenet5BW.LeNet5Binary,
    'lenet5XNOR': Lenet5XNOR.LeNet5XNOR,
    'lenet5Q': Lenet5Q.LenetQ,
    'alex': AlexNet.AlexNet,
}

def get_model(name):
    return MODEL_GETTER[name]