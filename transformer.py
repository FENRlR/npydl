"""
mlp training example with MNIST
"""
import numpy as np
import npydl as nd


nd.fix_seed()


# dataloader
class MNIST_DataLoader:
    """
    train-images-idx3-ubyte.gz: training set images, train-labels-idx1-ubyte.gz: training set labels
    t10k-images-idx3-ubyte.gz: test set images, t10k-labels-idx1-ubyte.gz: test set labels
    """
    def __init__(self, batch, train=True, shuffle=True, drop_last=False, dtype=np.float32, path="./mnist"):
        self.batch = batch
        self.shuffle = shuffle
        self.dtype = dtype
        self.drop_last = drop_last
        self.mode = 'train' if train is True else 'test'
        modestr = 'train' if train is True else 't10k'

        # data
        with open(path + fr"/{modestr}-images.idx3-ubyte", "rb") as f:
            raw1 = f.read()
        # info from the first 16bytes (4bytes -> [0]magic number, [1]num_items, [2]rows, [3]cols)
        header1 = np.frombuffer(raw1[:16], dtype='>i4')
        self.n_data, self.rows, self.cols = header1[1], header1[2], header1[3]
        self.data = np.frombuffer(raw1, dtype=np.uint8, offset=16) # starts from 16 bytes after
        self.data = self.data.reshape(self.n_data, self.rows * self.cols).astype(self.dtype) / 255.0

        # label
        with open(path + fr"/{modestr}-labels.idx1-ubyte", "rb") as f:
            raw2 = f.read()
        header2 = np.frombuffer(raw2[:8], dtype='>i4') # header -> 8 bytes (magic, num_items)
        n_labels = header2[1]
        self.labels = np.frombuffer(raw2, dtype=np.uint8, offset=8)
        self.labels = self.labels.reshape(n_labels, ).astype(np.int64)#.astype(self.dtype)

        self.cursor = 0

    def __len__(self):
        if self.drop_last:
            return self.n_data // self.batch
        else:
            return (self.n_data + self.batch - 1) // self.batch

    def shuffle_(self):
        indices = np.random.permutation(self.n_data)
        self.data = self.data[indices]
        self.labels = self.labels[indices]

    def enc_onehot(self, y, n_class=10): # [0, 9]
        out = np.zeros((len(y), n_class))
        out[np.arange(len(y)), y] = 1
        return out

    def __iter__(self):
        self.cursor = 0
        if self.shuffle:
            self.shuffle_()
        return self

    def __next__(self):
        if self.cursor >= self.n_data:
            raise StopIteration

        end = self.cursor + self.batch
        if end > self.n_data:
            if self.drop_last:
                raise StopIteration
            end = self.n_data

        batch_data = self.data[self.cursor:end]
        batch_labels = self.labels[self.cursor:end]
        self.cursor = end
        return batch_data, batch_labels


# model - transformer
class Model(nd.Mod):
    def __init__(self, num_layers, d_model, num_heads, d_ffn, bias=True, dtype=np.float32, requires_grad=True):
        super().__init__()
        self.tf_layers = num_layers
        self.layers = nd.ModList(
            *[TFB(d_model=d_model, num_heads=num_heads, d_ffn=d_ffn, bias=bias, dtype=dtype, requires_grad=requires_grad)
            for _ in range(num_layers)],
            nd.Linear(d_model, 10, bias=bias, dtype=dtype, requires_grad=requires_grad),
            nd.Softmax()
        )

    def fwd(self, x, mask=None):
        for i, layer in enumerate(self.layers):
            #print(x.shape)
            if i < self.tf_layers:
                x = layer.fwd(x, mask)
            else:
                x = layer.fwd(x)
        return x.reshape(x.shape[0],x.shape[-1])

    def bwd(self, dl):
        dl = dl.reshape(dl.shape[0], 1, dl.shape[-1])
        for i, layer in enumerate(reversed(self.layers)):
            #print(f"{i}, {x.shape}")
            dl = layer.bwd(dl)
        return dl


class TFB(nd.Mod): # transformer block
    def __init__(self, d_model, num_heads, d_ffn, bias=True, dtype=np.float32, requires_grad=True):
        super().__init__()
        self.mha = MHA(d_model, num_heads, bias=bias, dtype=dtype, requires_grad=requires_grad)
        self.norm1 = nd.LayerNorm(d_model)
        self.norm2 = nd.LayerNorm(d_model)
        self.ffn = FFN(d_model, d_ffn, bias=bias, dtype=dtype, requires_grad=requires_grad)

        self.add1 = nd.Matadd()
        self.add2 = nd.Matadd()

    def fwd(self, x, mask=None):
        attn = self.mha.fwd(x, mask)
        rsd1 = self.add1.fwd(x, attn)

        y = self.norm1.fwd(rsd1)
        y_ffn = self.ffn.fwd(y)

        rsd2 = self.add2.fwd(y, y_ffn)
        z = self.norm2.fwd(rsd2)
        return z

    def bwd(self, dl):
        dl = self.norm2.bwd(dl)
        dy_rsd2, dff = self.add2.bwd(dl)

        dy_ffn = self.ffn.bwd(dff)
        dy = dy_rsd2 + dy_ffn
        dy = self.norm1.bwd(dy)

        dx_rsd1, dattn = self.add1.bwd(dy)
        dx_attn = self.mha.bwd(dattn)

        dx = dx_rsd1 + dx_attn
        return dx


class MHA(nd.Mod):
    def __init__(self, d_model, num_heads=2, bias=True, dtype=np.float32, requires_grad=True):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.h = num_heads
        self.d_head = d_model // num_heads

        self.Wq = nd.Linear(d_model, d_model, bias=bias, dtype=dtype, requires_grad=requires_grad)
        self.Wk = nd.Linear(d_model, d_model, bias=bias, dtype=dtype, requires_grad=requires_grad)
        self.Wv = nd.Linear(d_model, d_model, bias=bias, dtype=dtype, requires_grad=requires_grad)
        self.Wo = nd.Linear(d_model, d_model, bias=bias, dtype=dtype, requires_grad=requires_grad)

        self.softmax = nd.Softmax()

        self.q = None
        self.k = None
        self.v = None
        self.attn = None
        self.scale = self.d_head ** 0.5

        self.mul_qk = nd.Matmul()
        self.mul_av = nd.Matmul()

    def hd_split(self, x):
        B, T, C = x.shape
        x = x.reshape(B, T, self.h, self.d_head)
        return x.transpose(0, 2, 1, 3)  # (B, h, T, d_head)

    def hd_combine(self, x):
        B, h, T, d = x.shape
        x = x.transpose(0, 2, 1, 3)
        return x.reshape(B, T, h * d)

    def fwd(self, x, mask=None):
        q = self.Wq.fwd(x)
        k = self.Wk.fwd(x)
        v = self.Wv.fwd(x)

        self.q = self.hd_split(q)
        self.k = self.hd_split(k)
        self.v = self.hd_split(v)

        score = self.mul_qk.fwd(self.q, self.k.transpose(0, 1, 3, 2))
        score = score / self.scale
        if mask is not None:
            score += mask

        self.attn = self.softmax.fwd(score)

        out = self.mul_av.fwd(self.attn, self.v)
        out = self.hd_combine(out)
        out = self.Wo.fwd(out)

        return out

    def bwd(self, dl):
        do = self.Wo.bwd(dl)
        do = self.hd_split(do)

        dattn, dv = self.mul_av.bwd(do)

        dscore = self.softmax.bwd(dattn)
        dscore /= self.scale

        dq, dkt = self.mul_qk.bwd(dscore)
        dk = dkt.transpose(0, 1, 3, 2)

        dq = self.hd_combine(dq)
        dk = self.hd_combine(dk)
        dv = self.hd_combine(dv)

        dx = self.Wq.bwd(dq)
        dx += self.Wk.bwd(dk)
        dx += self.Wv.bwd(dv)

        return dx

class FFN(nd.Mod):
    def __init__(self, dim, h_dim, bias=True, dtype=np.float32, requires_grad=True):
        super().__init__()
        self.bias = bias
        self.dtype = dtype
        self.requires_grad = requires_grad
        self.in_dim = dim
        self.h_dim = h_dim
        self.layers = nd.ModList(
            nd.Linear(self.in_dim, self.h_dim, bias=bias, dtype=dtype, requires_grad=requires_grad),
            nd.Relu(),
            nd.Linear(self.h_dim, self.in_dim, bias=bias, dtype=dtype, requires_grad=requires_grad)
        )

    def fwd(self, x): # forward
        for layer in self.layers:
            x = layer.fwd(x)
        return x

    def bwd(self, dl): # backward
        for layer in reversed(self.layers):
            dl = layer.bwd(dl)
        return dl


# train model
batch = 32
epoch = 5 # note : it would take some hours
dim = 784

trainloader = MNIST_DataLoader(batch, train=True, shuffle=True, drop_last=False, dtype=np.float32, path="./mnist")
testloader = MNIST_DataLoader(batch, train=False, shuffle=True, drop_last=False, dtype=np.float32, path="./mnist")
model = Model(num_layers=3, d_model=dim, num_heads=2, d_ffn=dim*2, dtype=np.float32, requires_grad=True) # (num_layers, d_model, num_heads, d_ffn, bias=True, dtype=np.float32, requires_grad=True)
optim = nd.Adam_W(params=model.params())
criterion = nd.CrossEntropyLoss()

_ = nd.load_ckpt(model, optim, "./transformer.npydl")

for e in range(epoch):
    # train
    total_loss = 0.0
    total_samples = 0
    for x, y in trainloader:
        #print(x.shape) # (32, 784)
        #print(y.shape) # (32, 1) -> 0~9 (n=10)
        optim.zero_grad()
        x = x.reshape(x.shape[0], 1, x.shape[-1])
        y_hat = model.fwd(x)
        loss = criterion.fwd(y_hat, y)

        dl = criterion.bwd() # dloss
        model.bwd(dl)
        optim.step()

        total_loss += loss * x.shape[0]
        total_samples += x.shape[0]
    tloss_avg = total_loss / total_samples
    #print(f"epoch {e} - loss : {tloss_avg}")

    # test
    val_loss = 0.0
    val_samples = 0
    pos = 0
    for x, y in testloader:
        x = x.reshape(x.shape[0], 1, x.shape[-1])
        y_hat = model.fwd(x)
        loss = criterion.fwd(y_hat, y)

        val_loss += loss * x.shape[0]
        val_samples += x.shape[0]
        pred = np.argmax(y_hat, axis=1)
        pos += np.sum(pred == y.reshape(-1))

    vloss_avg = val_loss / val_samples
    acc = pos / val_samples
    print(f"epoch {e+1} - train_loss: {tloss_avg:.4f}, val_loss: {vloss_avg:.4f}, acc: {acc:.4f}")

nd.save_ckpt(model, optim, epoch, f"./transformer.npydl")
