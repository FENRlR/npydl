# npydl
A lightweight NumPy-based deep learning framework for understanding how deep learning frameworks work.

## Features
- Custom autograd
- Neural network modules (Linear, Conv, Activations, etc.)
- Optimizers (SGD, Adam)
- Checkpoint save/load

## Usage
### - Model definition
```
import npydl as nd

class Model(nd.Mod):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.layers = nd.ModList(
            nd.Linear(in_dim, hidden_dim),
            nd.Relu(),
            nd.Linear(hidden_dim, hidden_dim),
            nd.Relu(),
            nd.Linear(hidden_dim, 10),
            nd.Softmax()
        )

    def fwd(self, x):
        for layer in self.layers:
            x = layer.fwd(x)
        return x

    def bwd(self, dl):
        for layer in reversed(self.layers):
            dl = layer.bwd(dl)
        return dl
```

### - Training loop
```
model = Model(in_dim, hidden_dim)
optim = nd.Adam_W(params=model.params())
criterion = nd.CrossEntropyLoss()
for x, y in trainloader:
    optim.zero_grad()
    
    y_hat = model.fwd(x)
    loss = criterion.fwd(y_hat, y)

    dl = criterion.bwd()
    model.bwd(dl)
    
    optim.step()
```

## Examples
- [MLP](./mlp.py)

- [Transformer](./transformer.py)
