from . import AlexNetQ, Lenet5, Lenet5BW, Lenet5XNOR, Lenet5Q, AlexNet



MODEL_GETTER = {
    'lenet5': Lenet5.LeNet5,
    'lenet5BW': Lenet5BW.LeNet5Binary,
    'lenet5XNOR': Lenet5XNOR.LeNet5XNOR,
    'lenet5Q': Lenet5Q.LenetQ,
    'alexQ': AlexNetQ.AlexNet,
    'alex': AlexNet.AlexNet
}

def get_model(name):
    return MODEL_GETTER[name]