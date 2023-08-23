import numpy as np
from engine import MicroTensor

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros(p.grad.shape)

    def parameters(self):
        return list()
    
class Layer(Module):

    def __init__(self, nin, nout, activation=None):
        self.dims = (nin, nout)
        self.W = MicroTensor(np.random.randn(nin, nout), 'W')
        self.b = MicroTensor(np.zeros((nout, 1)), 'b') 
        self.activation = activation

    def __call__(self, act_prev):
        act = self.W.T @ act_prev + self.b
        act = act.relu() if self.activation == 'relu' else act
        return act
    
    def parameters(self):
        return [self.W, self.b]
    
    def __repr__(self):
        return f'{"ReLU" if self.activation == "relu" else "Linear"}Layer{self.dims}'
    
class Sequential(Module):

    def __init__(self, *layers):
        self.layers = layers
        for idx, layer in enumerate(self.layers):
            layer.W.data *= np.sqrt(2/layer.dims[0]) if layer.activation == 'relu' else 0.01 
            layer.W.name, layer.b.name = (f'W{idx+1}', f'b{idx+1}')

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
    def __repr__(self):
        return f"Sequential({ f', '.join(str(layer) for layer in self.layers)})"