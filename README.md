# mini-grad

Inspired by : [GeoHot's TinyGrad](https://github.com/geohot/tinygrad)

<p align='justify'>
Learning how to make a custom deep learning framework using only numpy.
Mostly, for now I will try to understand what George did in his work and recreate the same results.
Hopefully, I will be able to add something new to it. If not, I will have at least learned something.
</p>

### Example

```python
import numpy as np
from minigrad.tensor import Tensor

x = Tensor(np.eye(3))
y = Tensor(np.array([[2.0,0,-2.0]]))
z = y.dot(x).sum()
z.backward()

print(x.grad)  # dz/dx
print(y.grad)  # dz/dy
```


### Same example in torch

```python
import torch

x = torch.eye(3, requires_grad=True)
y = torch.tensor([[2.0,0,-2.0]], requires_grad=True)
z = y.matmul(x).sum()
z.backward()

print(x.grad)  # dz/dx
print(y.grad)  # dz/dy
```

### Neural Networks ??
<p align='justify'>
It turns out, a decent autograd tensor library is 90% of what you need for neural networks. Add an optimizer (SGD and Adam implemented) from minigrad.optim, write some boilerplate minibatching code, and you have all you need.
</p>

### TODO (to make real neural network library)

* Look into Adam Optimizer
* Implement convolutions
