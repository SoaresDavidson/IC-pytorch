import torch
import torch.nn as nn
import torch.nn.functional as F

class BinFunction(torch.autograd.Function): #função de ativação
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output): # type: ignore
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input

class Binarize(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, input):
        return BinFunction.apply(input)


class BinConvInput(torch.autograd.Function): #binarização da imagem com calculo do beta[incompleto]
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        temp = input.clone()
        _, features,width,height = input.size()

        alpha = temp.abs().mean(1, keepdim=True).expand_as(input).clone() 

        k = torch.ones(width , height, device=input.device).div(width*height)
        k = k.expand(features,1,width,height)

        beta = F.conv2d(alpha, k, groups=features)
        return beta * Binarize()(input)

    @staticmethod
    def backward(ctx, grad_output): #type: ignore
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input
        # grad_bin = grad_output.sign()
        # scaling = grad_output.amax((1,2,3), keepdim=True)

        # return grad_bin * scaling

class BinarizeAct(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, input):
        return BinConvInput.apply(input)

class BinWeights(torch.autograd.Function): #binarização dos pesos com STE
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.sign(x)
    
    @staticmethod
    def backward(ctx, grad_output): # type: ignore #1 back 2 back
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input
    
class BinarizeWeights(nn.Module):
    def __init__(self):
        super(BinarizeWeights, self).__init__()

    def forward(self, input):
        return BinWeights.apply(input)
    
class BinConvParam(torch.autograd.Function): #binarização da camada convolucional
    @staticmethod
    def forward(ctx, weights):
        ctx.save_for_backward(weights)
        mean_weight = weights - weights.mean(1, keepdim=True).expand_as(weights)

        clamped_weight = mean_weight.clamp(-1.0, 1.0)

        alpha = clamped_weight.abs()\
                .mean((1,2,3), keepdim=True)\
                .expand_as(weights)

        return torch.sign((clamped_weight).mul(alpha))

    @staticmethod
    def backward(ctx, grad_output): #type: ignore 3 back 4 back
        weights, = ctx.saved_tensors
        num_weights = weights.numel()

        alpha = weights.abs()\
                .mean((1,2,3), keepdim=True)\
                .expand_as(weights).clone()
        
        alpha[weights.lt(-1.0)] = 0
        alpha[weights.gt(1.0)] = 0

        alpha_grad = alpha.mul(grad_output)

        mean_grad = grad_output.div(num_weights)

        return mean_grad + alpha_grad #type: ignore
class Conv2dBinary(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, bias=bias
        )
    
    def forward(self, input):
        weight_binarized = BinConvParam.apply(self.weight)

        output = F.conv2d(
            input, weight_binarized, self.bias, self.stride, self.padding, self.dilation, self.groups #type: ignore
        ) 
        return output

class BinLinearParam(torch.autograd.Function): # binarização da camada linear
    @staticmethod
    def forward(ctx, weights) -> torch.Tensor:
        ctx.save_for_backward(weights)
        mean_weight = weights - weights.mean(1, keepdim=True).expand_as(weights)
        
        clamped_weight = mean_weight.clamp(-1.0, 1.0)

        alpha = clamped_weight.abs()\
                .mean(1, keepdim=True)\
                .expand_as(weights)
        
        return torch.sign((clamped_weight).mul(alpha))
    
    @staticmethod
    def backward(ctx, grad_output): #type: ignore 5 back 6 back
        weights, = ctx.saved_tensors
        num_weights = weights.numel()

        alpha = weights.abs().mean(1,keepdim=True).expand_as(weights).clone()

        alpha[weights.gt(1.0)] = 0
        alpha[weights.lt(-1.0)] = 0

        alpha_grad = alpha.mul(grad_output)

        mean_grad = grad_output.div(num_weights)

        return mean_grad + alpha_grad #type: ignore
class LinearBinary(nn.Linear):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__(in_features, out_features, bias=bias)
        self.binarize = BinLinearParam.apply

    def forward(self, input):
        self.weight.copy_(self.binarize(self.weight))

        output = F.linear(input, self.weight, self.bias) 
        return output
    
class C3(nn.Module):
    def __init__(self):
        super().__init__()

        self.feature_maps = [
            [0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [0, 4, 5], [0, 1, 5],
            [0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [0, 3, 4, 5], [0, 1, 4, 5], [0, 1, 2, 5],
            [0, 1, 3, 4], [1, 2, 4, 5], [0, 2, 3, 5],
            [0, 1, 2, 3, 4, 5]
        ]

        self.conv_layer = nn.ModuleList()

        for input in self.feature_maps:
            conv = nn.Conv2d(in_channels=len(input),out_channels=1,kernel_size=5)
            self.conv_layer.append(conv)
        
    def forward(self, input):
        c3 = []
        for i, maps in enumerate(self.feature_maps):

            S2 = input[:, maps, :,:]
            resultado = self.conv_layer[i](S2)
            c3.append(resultado)

        return torch.cat(c3, dim=1)
    
def uniform_quantize(k):
  class qfn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
      if k == 32:
        out = input
      elif k == 1:
        out = torch.sign(input)
      else:
        n = float(2 ** k - 1)
        out = torch.round(input * n) / n
      return out

    @staticmethod
    def backward(ctx, grad_output):
      grad_input = grad_output.clone()
      return grad_input

  return qfn().apply
class weight_quantize_fn(nn.Module):
  def __init__(self, w_bit):
    super(weight_quantize_fn, self).__init__()
    assert w_bit <= 8 or w_bit == 32
    self.w_bit = w_bit
    self.uniform_q = uniform_quantize(k=w_bit)

  def forward(self, x):
    if self.w_bit == 32:
      weight_q = x
    elif self.w_bit == 1:
      E = torch.mean(torch.abs(x)).detach()
      weight_q = self.uniform_q(x / E) * E
    else:
      weight = torch.tanh(x)
      max_w = torch.max(torch.abs(weight)).detach()
      weight = weight / 2 / max_w + 0.5
      weight_q = max_w * (2 * self.uniform_q(weight) - 1)
    return weight_q
  

class activation_quantize_fn(nn.Module):
  def __init__(self, a_bit):
    super(activation_quantize_fn, self).__init__()
    assert a_bit <= 8 or a_bit == 32
    self.a_bit = a_bit
    self.uniform_q = uniform_quantize(k=a_bit)

  def forward(self, x):
    if self.a_bit == 32:
      activation_q = x
    else:
      activation_q = self.uniform_q(torch.clamp(x, 0, 1))
      # print(np.unique(activation_q.detach().numpy()))
    return activation_q
  
class Conv2dQ(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, k_bits = 8):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, bias=bias
        )
        self.quantize_w = weight_quantize_fn(w_bit=k_bits)
    
    def forward(self, input):
        weight_quantized = self.quantize_w(self.weight)
        output = F.conv2d(
            input, weight_quantized, self.bias, self.stride, self.padding, self.dilation, self.groups #type: ignore
        ) 
        return output

class LinearQ(nn.Linear):
    def __init__(self, in_features, out_features, bias=False, k_bits = 8):
        super().__init__(in_features, out_features, bias=bias)
        self.quantize_w = weight_quantize_fn(w_bit=k_bits)

    def forward(self, input):
        weight_quantized = self.quantize_w(self.weight)
        output = F.linear(input, weight_quantized, self.bias) #type: ignore
        return output
   
class fg(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, k):
        ctx.save_for_backward(input)
        ctx.k = k
        return input
    
    @staticmethod
    def backward(ctx, *grad_outputs):
        input, = ctx.saved_tensors
        k = ctx.k
        grad = grad_outputs[0]

        epsilon = 1e-8
        max_abs = grad.abs().max()
        scale = max(epsilon, max_abs)

        grad_norm = grad / (2*scale)
        
        sigma = torch.empty_like(grad).uniform_(-0.5, 0.5)
        N = sigma/(2 ** k - 1)

        input_q = grad_norm + 0.5 + N
        quant = uniform_quantize(k)(input_q.clamp(0, 1))

        grad_quantized = (2 * scale) * (quant - 0.5)

        return grad_quantized, None
    
class QuantizeGradient(nn.Module):
    def __init__(self, k_bits):
        super().__init__()
        self.k = k_bits

    def forward(self, x):
        return fg.apply(x, self.k)





# @tf.custom_gradient
#     def _identity(input):
#         def grad_fg(x):
#             rank = x.get_shape().ndims
#             assert rank is not None
#             maxx = tf.reduce_max(tf.abs(x), list(range(1, rank)), keep_dims=True)
#             x = x / maxx
#             n = float(2**bitG - 1)
#             x = x * 0.5 + 0.5 + tf.random_uniform(
#                 tf.shape(x), minval=-0.5 / n, maxval=0.5 / n)
#             x = tf.clip_by_value(x, 0.0, 1.0)
#             x = quantize(x, bitG) - 0.5
#             return x * maxx * 2

#         return input, grad_fg

#     return _identity(x)
