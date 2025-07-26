import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F



class BinOp():
    def __init__(self, model): #pega as camadas que vão ter os pesos binarizados
        # count the number of Conv2d and Linear
        self.saved_params = []
        self.target_modules = []
        first_layer = True
        modules = list(model.modules())
        for idx, m in enumerate(modules):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                if first_layer or idx == len(modules)-1:
                    first_layer = False
                    continue
                tmp = m.weight.detach().clone()
                self.saved_params.append(tmp) #salva os pesos
                self.target_modules.append(m.weight)

        return

    def binarization(self): #faz tudo
        self.meancenterConvParams()
        self.clampConvParams()
        self.save_params()
        self.binarizeConvParams()

    def meancenterConvParams(self): #calcula a média absoluta dos pesos e subtrai dele mesmo (centraliza os pesos)
        for param in self.target_modules:
            # s = self.target_modules[index].data.size() não é utilizado
            with torch.no_grad():
                negMean = param.mean(1, keepdim=True).mul(-1).expand_as(param)
                param.add_(negMean)

    def clampConvParams(self): #entender depois mas basicamente limita os  pesos para 1 e -1
        for param in self.target_modules:
            with torch.no_grad():
                param.copy_(param.clamp(-1.0, 1.0))

    def save_params(self): #copia os pesos em target_module para saved_params
        for index, target in enumerate(self.target_modules):
            with torch.no_grad():
                self.saved_params[index].copy_(target) #copia inplace target_module para saved_params

    def binarizeConvParams(self): # calcula o alpha 
        with torch.no_grad():
            for param in self.target_modules:
                dim = param.size() #dimensões
                # print(f"dimensão:{dim}")
                if len(dim) == 4: #conlucional = [saída, entrada, altura_kernel, largura_kernel]
                    alpha = param.abs().mean(dim=(1,2,3), keepdim=True).expand(dim) #média dos valores absolutos
                elif len(dim) == 2: #linear = [entrada, saída]
                    alpha = param.abs().mean(dim=1, keepdim=True).expand(dim)

                param.copy_(param.sign().mul(alpha)) #type: ignore

    def restore(self):
        for idx, param in enumerate(self.saved_params):
            with torch.no_grad():
                self.target_modules[idx].copy_(param) #target_module = saved_param

    def updateBinaryGradWeight(self):
        with torch.no_grad():
            for param in self.target_modules:
                saidas = param[0].nelement() #num de saídas
                dim = param.size() #dimensões
                # print(f"saídas:{saidas}")
                # print(f"dimensão:{dim}")
                if len(dim) == 4:
                    alpha = param.abs()\
                            .mean(dim=(1,2,3), keepdim=True)\
                            .expand(dim).clone()
                elif len(dim) == 2:
                    alpha = param.abs().mean(1, keepdim=True).expand(dim).clone()

                alpha[param.lt(-1.0)] = 0 #type: ignore
                alpha[param.gt(1.0)] = 0 #type: ignore

                alpha.mul_(param.grad) #type: ignore #alpha * gradiente dos pesos
                
                alpha_add = param.grad.div(saidas)

                param.grad = alpha.add(alpha_add).mul(1.0-1.0/dim[1]) #type: ignore 
                #.mul(1.0-1.0/s[1]) heuristica: input plane scaling



def plot_batch(dataloader, classes):
    data_iter = iter(dataloader)
    images, labels = next(data_iter)
    # title=' '.join(classes[label] for label in labels)
    npimg = torchvision.utils.make_grid(images).numpy()
    plt.figure(figsize=(10,4))
    plt.imshow(np.transpose(npimg, (1,2,0)))
    # plt.title(title)
    plt.axis('off')
    plt.show()


def images_to_probs(output):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.cpu().numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


# def plot_classes_preds(output, images, labels):

#     preds, probs = images_to_probs(output, images)
#     # plot the images in the batch, along with predicted and true labels
#     fig = plt.figure(figsize=(48, 48))
#     for idx in np.arange(64):
#         ax = fig.add_subplot(1, 64, idx+1, xticks=[], yticks=[])
#         plt.imshow(images[idx].squeeze().numpy(), cmap='gray')
#         ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
#             labels[preds[idx]],
#             probs[idx] * 100.0,
#             labels[labels[idx]]),
#                     color=("green" if preds[idx]==labels[idx].item() else "red"))
#     return fig

def plot_classes_preds(output, images, labels):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    label_names = ['0', '1', '2', '3', '4',
               '5', '6', '7', '8', '9']
    preds, probs = images_to_probs(output)
    n_images = min(len(images), 64)

    rows = 8
    cols = 8
    fig = plt.figure(figsize=(cols * 2, rows * 2))  # adjust size for visibility

    for idx in range(n_images):
        ax = fig.add_subplot(rows, cols, idx + 1, xticks=[], yticks=[])
        image = images[idx].detach().cpu().squeeze().numpy()
        plt.imshow(image, cmap='gray')

        pred = preds[idx]
        prob = probs[idx]
        label = labels[idx].item()

        ax.set_title(
            f"{label_names[pred]}\n{prob:.1%}\n(label: {label_names[label]})",
            fontsize=6,
            color=("green" if pred == label else "red")
        )
    fig.tight_layout()
    # fig.savefig("prediction.png")
    return fig
