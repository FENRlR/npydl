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


# model - mlp
class Model(nd.Mod):
    def __init__(self, in_dim, hidden_dim, bias=True, dtype=np.float32, requires_grad=True):
        super().__init__()
        self.bias = bias
        self.dtype = dtype
        self.requires_grad = requires_grad

        self.in_dim = in_dim
        self.h_dim = hidden_dim
        self.layers = nd.ModList(
            nd.Linear(self.in_dim, self.h_dim, bias=bias, dtype=dtype, requires_grad=requires_grad),
            nd.Relu(),
            nd.Linear(self.h_dim, self.in_dim, bias=bias, dtype=dtype, requires_grad=requires_grad),
            nd.Relu(),
            nd.Linear(self.in_dim, 10, bias=bias, dtype=dtype, requires_grad=requires_grad),
            nd.Softmax()
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
epoch = 10
dim = 784

trainloader = MNIST_DataLoader(batch, train=True, shuffle=True, drop_last=False, dtype=np.float32, path="./mnist")
testloader = MNIST_DataLoader(batch, train=False, shuffle=True, drop_last=False, dtype=np.float32, path="./mnist")
model = Model(in_dim=dim, hidden_dim=dim*2, requires_grad=True)
optim = nd.Adam_W(params=model.params())
criterion = nd.CrossEntropyLoss()

_ = nd.load_ckpt(model, optim, "./mlp.npydl")

for e in range(epoch):
    # train
    total_loss = 0.0
    total_samples = 0
    for x, y in trainloader:
        #print(x.shape) # (32, 784)
        #print(y.shape) # (32, 1) -> 0~9 (n=10)
        optim.zero_grad()
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
        y_hat = model.fwd(x)
        loss = criterion.fwd(y_hat, y)

        val_loss += loss * x.shape[0]
        val_samples += x.shape[0]
        pred = np.argmax(y_hat, axis=1)
        pos += np.sum(pred == y.reshape(-1))

    vloss_avg = val_loss / val_samples
    acc = pos / val_samples
    print(f"epoch {e+1} - train_loss: {tloss_avg:.4f}, val_loss: {vloss_avg:.4f}, acc: {acc:.4f}")

nd.save_ckpt(model, optim, epoch, f"./mlp.npydl")
