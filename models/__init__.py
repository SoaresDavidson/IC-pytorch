from . import Lenet5, Lenet5BW, Lenet5Quant, Lenet5XNOR, AlexNet, AlexNetCifar100



MODEL_GETTER = {
    'lenet5': Lenet5.LeNet5,
    'lenet5BW': Lenet5BW.LeNet5Binary,
    # 'lenet5Q': Lenet5Quant.Lenet5Quant,
    'lenet5XNOR': Lenet5XNOR.LeNet5XNOR,
    'alex': AlexNet.AlexNet,
    'alex2': AlexNetCifar100.AlexNet2
}

def get_model(name):
    return MODEL_GETTER[name]