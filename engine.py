import numpy as np 
from contextlib import contextmanager

class MicroTensor:
    """ 
        MicroTensor Engine
        Wrapper class encapsulating matrix functions. Tracks operations 
        and allows for backpropagation / gradient calculation.

        MicroTensor is a matrix-based version of Andrej Karpathy's scalar-based Micrograd (https://github.com/karpathy/micrograd)

        Aug. 2023 Theo Nakfoor
    """

    _trackGrad = True

    def __init__(self, data, name, _children=()):
        self.data = data.astype(float)
        self.shape = self.data.shape
        self.name = name
        self.grad = np.zeros(data.shape)
        self._backward = lambda: None
        self._prev = tuple(_children)

    def __add__(self, other):
        other = other if isinstance(other, MicroTensor) else MicroTensor(np.asarray(other), 'unk')
        out = MicroTensor(np.add(self.data, other.data), f'({self.name}+{other.name})', (self, other) if (self._trackGrad and other._trackGrad) else ())

        def _backward():
            self.grad = self.grad + (np.sum(out.grad, axis=1, keepdims=True)/out.grad.shape[1]) # Normalize over mini-batch size. NOTE: Test more
            other.grad = other.grad + (np.sum(out.grad, axis=1, keepdims=True)/out.grad.shape[1])
        out._backward = _backward if (self._trackGrad and other._trackGrad) else lambda: None

        return out
    
    def __matmul__(self, other):
        other = other if isinstance(other, MicroTensor) else MicroTensor(np.asarray(other), 'unk')
        out = MicroTensor(np.dot(self.data, other.data), f'{self.name}@{other.name}', (self, other) if (self._trackGrad and other._trackGrad) else ())

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += (self.data.T @ out.grad) / self.data.shape[0]
        out._backward = _backward if (self._trackGrad and other._trackGrad) else lambda: None

        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, MicroTensor) else MicroTensor(np.asarray(other), 'unk')
        out = MicroTensor(np.multiply(self.data, other.data), f'{self.name}*{other.name}', (self, other) if(self._trackGrad and other._trackGrad) else ())

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward if (self._trackGrad and other._trackGrad) else lambda: None

        return out

    def relu(self):
        temp = np.copy(self.data)
        temp[temp < 0] = 0
        out = MicroTensor(temp.astype(float), f'ReLU({self.name})', (self,) if self._trackGrad else ())

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward if self._trackGrad else lambda: None

        return out 
    
    @property
    def T(self):
        temp = np.copy(self.data)
        out = MicroTensor(temp.T, f'{self.name}.T', (self,) if self._trackGrad else ())

        def _backward():
            self.grad += out.grad.T
        out._backward = _backward if self._trackGrad else lambda: None

        return out

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        other = other if isinstance(other, MicroTensor) else MicroTensor(np.asarray(other), 'unk')
        return other + self
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        other = other if isinstance(other, MicroTensor) else MicroTensor(np.asarray(other), 'unk')
        return other + (-self)
    
    def __rmul__(self, other):
        other = other if isinstance(other, MicroTensor) else MicroTensor(np.asarray(other), 'unk')
        return other * self
    
    def backward(self):
        children = list()
        def traverse(x):
            if x not in children:
                for p in x._prev:
                    traverse(p)
                children.append(x)
        traverse(self)
        self.grad = np.ones(self.data.shape)
        for c in reversed(children):
            c._backward()

    @property
    def trace(self):
        return f'<{self.name}>'

    @contextmanager 
    def track_grad(state):
        _prev = MicroTensor._trackGrad
        try:
            MicroTensor._trackGrad = state
            yield
        finally:
            MicroTensor._trackGrad = _prev

    def __repr__(self):
        return f'MicroTensor(trace=<{self.name}>, shape={self.data.shape}, data={self.data}, grad={self.grad})'