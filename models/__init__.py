from . import AlexNetQ, Lenet5, Lenet5XNOR, Lenet5Q, AlexNet, AlexNetXNOR, NiN, residual



MODEL_GETTER = {
    'lenet5': Lenet5.LeNet5,
    'lenet5XNOR': Lenet5XNOR.LeNet5XNOR,
    'lenet5Q': Lenet5Q.LenetQ,
    'alex': AlexNet.AlexNet,
    'alexX': AlexNetXNOR.AlexNet,
    'alexQ': AlexNetQ.AlexNet,
    'nin': NiN.NiN,
    'resnet': residual.ResNet
}

def get_model(name):
    return MODEL_GETTER[name]