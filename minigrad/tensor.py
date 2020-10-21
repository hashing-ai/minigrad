#!/usr/bin/env python3

from functools import partialmethod

import numpy as np

# Let's create a Tensor Class

class Tensor:
    def __init__(self,data):
        # Error Message if data is not np.ndarray
        assert (type(data) == np.ndarray), "Check the input and make sure it's a numpy array."

        self.data = data
        self.grad = None

        # internal variables used for autograd graph construction
        self._ctx = None

    def __repr__(self):
        return "Tensor data : %r with Gradient : %r" % (self.data, self.grad)

    def backward(self, initialize=True):

        if self._ctx is None:
            return

        if self.grad is None and initialize:
            # filling in the first gradient with one
            # called "implicit gradient creation"
            assert self.data.size == 1
            self.grad = np.ones_like(self.data)

        assert(self.grad is not None)

        grads = self._ctx.backward(self._ctx, self.grad)

        if len(self._ctx.parents) == 1:
            grads = [grads]
        for t,g in zip(self._ctx.parents, grads):
            assert (g.shape == t.data.shape), ("Gradient shape must match Tensor shape in %r, %r != %r") % (self._ctx, g.shape, t.data.shape)
            t.grad = g
            t.backward(False)

    def mean(self):
        div = Tensor(np.array([1/self.data.size]))
        return self.sum().mul(div)


# An instantiation of the Function is the context
class Function:
    def __init__(self, *tensors):
        self.parents = tensors
        self.saved_tensors = []

    def save_for_backward(self, *x):
        self.saved_tensors.extend(x)

    # because of partialmethod usage :: need to support the args in both order
    # partialmethod(self, *arg_to_freeze) or partialmethod(*arg_to_freeze, self)
    def apply(self, arg, *x):
        if type(arg) == Tensor:
            op = self
            x = [arg]+list(x)
        else:
            op = arg
            x = [self]+list(x)
        ctx = op(*x)
        ret = Tensor(op.forward(ctx, *[t.data for t in x]))
        ret._ctx = ctx
        return ret

    def register(name, fxn):
        setattr(Tensor, name, partialmethod(fxn.apply,fxn))



# ----- implement functions ----- @

class Mul(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x,y)
        return x*y

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        return y*grad_output, x*grad_output
Function.register('mul', Mul)


class Add(Function):
    @staticmethod
    def forward(ctx, x, y):
        return x+y

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_output
Function.register('add', Add)


class ReLU(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return np.maximum(input,0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.copy()
        grad_input[input < 0] = 0
        return grad_input
Function.register('relu', ReLU)


class Dot(Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        return input.dot(weight)

    @ staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        grad_input = grad_output.dot(weight.T)
        grad_weight = grad_output.T.dot(input).T
        return grad_input, grad_weight
Function.register('dot',Dot)

class Sum(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return np.array([input.sum()])

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output * np.ones_like(input)
Function.register('sum', Sum)

class LogSoftmax(Function):
    @staticmethod
    def forward(ctx, input):
        def logsumexp(x):
            c = x.max(axis=1)
            return c +  np.log(np.exp(x-c.reshape(-1,1)).sum(axis=1))

        output = input - logsumexp(input).reshape((-1,1))
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        return grad_output - np.exp(output)*grad_output.sum(axis=1).reshape((-1,1))
Function.register('logsoftmax', LogSoftmax)



