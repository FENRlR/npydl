import numpy as np
import math
import os


# random seed
def fix_seed(x=1557):
    np.random.seed(x)


# init dist
def xavier(fin, fout):
    scale = 1 / max(1., (fin + fout) / 2.)
    limit = math.sqrt(3.0 * scale) # where limit is sqrt(6 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, size=(fin, fout))


# func
def sigmoid(x):
    return 1/(1+np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    x = np.exp(x - np.max(x)) # overflow proof
    return x / np.sum(x)


# loss func
def sum_squares_loss(y, tgt): # L2
    return np.sum((y-tgt)**2) * 0.5


def cross_entropy_error(y, tgt):
    y, tgt = (y.reshape(1, y.size), tgt.reshape(1, tgt.size)) if y.ndim == 1 else (tgt, y)
    return -np.sum(tgt * np.log(y + 1e-9)) / y.shape[0]


# generic components
class Parameter:
    def __init__(self, *shape, init='xavier', dtype=np.float32, requires_grad=True):
        self.shape = shape # [in_dim, ..., out_dim]
        self.req_grad = requires_grad
        self.grad = None
        if init == 'xavier':
            if len(shape) == 1:
                in_dim = shape[0]
                out_dim = shape[0]
            elif len(shape) == 2:
                in_dim, out_dim = shape # self.shape = (in_dim, out_dim)
            else:
                inter = np.prod(shape[1:-1])
                in_dim = shape[0] * inter
                out_dim = shape[-1] * inter
            self.mat = xavier(in_dim, out_dim).astype(dtype)
        #if init == 'xavier':
        #    self.mat = xavier(in_dim, out_dim).astype(dtype)
        elif init == 'zeros':
            self.mat = np.zeros(self.shape).astype(dtype)
        elif init == 'ones':
            self.mat = np.ones(self.shape).astype(dtype)


class Mod: # generic module
    def __init__(self):
        self._modules_ = {}
        self._parameters_ = {}

    def __setattr__(self, name, value):
        if isinstance(value, Mod):
            self._modules_[name] = value
        elif isinstance(value, Parameter):
            self._parameters_[name] = value
        object.__setattr__(self, name, value)

    def params(self):
        parameters = []
        parameters += list(self._parameters_.values())
        for m in self._modules_.values():
            parameters += m.params()
        return parameters

    #def modval(self):
    #    return self._modules_.values()


# operator
class Matmul:
    def __init__(self):
        self.x = None
        self.w = None
    def fwd(self, x, w):
        self.x = x
        self.w = w
        return np.dot(x, w) # y = xw
    def bwd(self, dl):
        dx = np.dot(dl, self.w.T) # dL/dx = dL/dy * dy/dx
        dw = np.dot(self.x.T, dl)
        return dx, dw


class Matadd:
    def __init__(self):
        self.x_shape = None
        self.y_shape = None
    def fwd(self, x, y):
        self.x_shape = x.shape
        self.y_shape = y.shape
        return x + y
    def bwd(self, dl):
        dx = dl # dl * 1
        dy = dl # dl * 1
        if dx.shape != self.x_shape: # for broadcasting
            dx = np.sum(dx, axis=0, keepdims=True)
        if dy.shape != self.y_shape:
            dy = np.sum(dy, axis=0, keepdims=True)
        return dx, dy


# func
class Relu(Mod): # np.maximum(0, x)
    def __init__(self):
        super().__init__()
        self.mask = None

    def fwd(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def bwd(self, dl):
        dl[self.mask] = 0
        dx = dl
        return dx


class Sigmoid(Mod):
    def __init__(self):
        super().__init__()
        self.out = None

    def fwd(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def bwd(self, dl):
        dx = dl * (1.0 - self.out) * self.out
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None # from softmax
        self.t = None # tlabel one-hot

    def fwd(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def bwd(self):
        dx = (self.y - self.t) / self.t.shape[0]
        return dx


class Softmax(Mod):
    def __init__(self, axis=-1):
        super().__init__()
        self.axis = axis
        self.out = None

    def fwd(self, x):
        x_rel = x - np.max(x, axis=self.axis, keepdims=True)
        x_exp = np.exp(x_rel)
        out = x_exp / np.sum(x_exp, axis=self.axis, keepdims=True)
        self.out = out
        return out

    def bwd(self, dl):
        y = self.out
        dot = np.sum(dl * y, axis=self.axis, keepdims=True)
        dx = y * (dl - dot)
        return dx


class CrossEntropyLoss: # -np.sum(tgt * np.log(y + 1e-9)) / y.shape[0]
    def __init__(self, eps=1e-9):
        self.probs = None
        self.labels = None
        self.eps = eps

    def fwd(self, probs, labels): # softmax output: (B, C), labels: (B,)
        self.probs = probs
        self.labels = labels

        B = probs.shape[0]
        loss = -np.log(probs[np.arange(B), labels] + self.eps)
        return np.mean(loss)

    def bwd(self): # dl/dprobs
        B = self.probs.shape[0]
        dx = np.zeros_like(self.probs)

        dx[np.arange(B), self.labels] = -1 / (self.probs[np.arange(B), self.labels] + self.eps)
        dx /= B

        return dx


# layer
class Dropout(Mod):
    def __init__(self, dropout_ratio=0.5):
        super().__init__()
        self.d_prop = dropout_ratio
        self.mask = None

    def fwd(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.d_prop
            return x * self.mask
        else:
            return x * (1.0 - self.d_prop)

    def bwd(self, dl):
        return dl * self.mask


class RMSNorm(Mod):
    def __init__(self, feature_dim, eps=1e-8):
        super().__init__()
        self.dim = feature_dim
        self.eps = eps
        self.r = Parameter(1, self.dim, init='ones')
        self.rms = None
        self.x_hat = None

    def fwd(self, x): # x shape: [..., F]
        self.rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + self.eps)
        self.x_hat = x / self.rms
        return self.x_hat * self.r.mat # broadcasting

    def bwd(self, dl):
        self.r.grad = np.sum(dl * self.x_hat, axis=tuple(range(dl.ndim - 1))).reshape(self.r.mat.shape) # dl/dy * x/rms -> [1, f], sums all but feat dim
        dx_hat = dl * self.r.mat
        dx = (dx_hat - self.x_hat * np.mean(dx_hat * self.x_hat, axis=-1, keepdims=True)) / self.rms
        return dx


class Linear(Mod):
    def __init__(self, in_dim, out_dim, bias=True, dtype=np.float32, requires_grad=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.w = Parameter(in_dim, out_dim, dtype=dtype, requires_grad=requires_grad)
        if bias:
            self.b = Parameter(1, out_dim, dtype=dtype, requires_grad=requires_grad)
        else:
            self.b = None

        self.req_grad = requires_grad
        self.mul = Matmul()
        self.add = Matadd()

    def fwd(self, x):
        y = self.mul.fwd(x, self.w.mat)  # x @ W
        if self.b is not None:
            y = self.add.fwd(y, self.b.mat) # + b
        return y

    def bwd(self, dl):
        if self.b is not None:
            dl, db = self.add.bwd(dl)
        dx, dw = self.mul.bwd(dl)
        if self.req_grad:
            if self.w.grad is None:
                self.w.grad = dw
            else:
                self.w.grad += dw
            if self.b is not None:
                if self.b.grad is None:
                    self.b.grad = db
                else:
                    self.b.grad += db
        return dx


def im2col(x, out_h, out_w, FH, FW, stride=1): # trans - 2dim
    N, _, H, W = x.shape # x: [N, C, H, W]
    cols = []
    for i in range(0, H - FH + 1, stride): # height -> move by stride
        for j in range(0, W - FW + 1, stride): # width -> stride
            col = x[:, :, i:i+FH, j:j+FW].reshape(N, -1)  # patch -> [N, C*FH*FW]
            cols.append(col) # list
    cols = np.stack(cols, axis=1)  # [N, out_h*out_w, C*FH*FW]
    cols = cols.reshape(N*out_h*out_w, -1)  # 'linear-like' -> [N*out_h*out_w, C*FH*FW]
    return cols


def col2im(cols, x_shape, out_h, out_w, FH, FW, stride=1): # restore - dl/dx -> shape [N,C,H,W]
    N, C, H, W = x_shape
    cols = cols.reshape(N, out_h*out_w, -1)  # [N, out_h*out_w, C*FH*FW]
    x = np.zeros(x_shape, dtype=cols.dtype)
    crt = 0 # current pos index for sliding window
    for i in range(0, H - FH + 1, stride):
        for j in range(0, W - FW + 1, stride):
            x[:, :, i:i+FH, j:j+FW] += cols[:, crt, :].reshape(N, C, FH, FW)
            crt += 1
    return x


class Conv2D(Mod):
    def __init__(self, in_dim, out_dim, *kernel_dim, stride=1, pad=0, bias=True, dtype=np.float32, requires_grad=True):
        super().__init__()
        if len(kernel_dim) == 1:
            self.fh = kernel_dim[0]
            self.fw = kernel_dim[0]
        elif len(kernel_dim) == 2:
            self.fh, self.fw = kernel_dim
        self.in_dim = in_dim # ch
        self.out_dim = out_dim # flt
        # input[c h w] * w[c fh fw] = [fn oh ow] ([1 oh ow] -> iter fn)
        self.w = Parameter(in_dim, self.fh, self.fw, out_dim, dtype=dtype, requires_grad=requires_grad) #-! PARAM
        if bias:
            self.b = Parameter(1, out_dim, dtype=dtype, requires_grad=requires_grad)
        else:
            self.b = None
        self.stride = stride
        self.pad = pad
        self.req_grad = requires_grad
        self.mul = Matmul()
        self.add = Matadd()

        self.xp_shape = None # x_pad.shape
        self.oh = None
        self.ow = None

    def fwd(self, x):
        B, C, H, W = x.shape # x: [N, C, H, W]
        x_pad = np.pad(x, ( (0, 0), (0, 0), (self.pad, self.pad), (self.pad, self.pad) ), mode='constant', constant_values=0) # padding
        self.xp_shape = x_pad.shape

        self.oh = (H + 2 * self.pad - self.fh) // self.stride + 1 # out_h = (H - FH) // stride + 1
        self.ow = (W + 2 * self.pad - self.fw) // self.stride + 1

        x_col = im2col(x_pad, self.oh, self.ow, self.fh, self.fw, self.stride)  # [N * oh * ow, C * fh * fw] # im2col trans
        w_col = self.w.mat.reshape(-1, self.out_dim)  # [C*fh*fw, out_dim] # weight reshape
        y = self.mul.fwd(x_col, w_col)  # [N*out_h*out_w, out_dim] # conv via matmul
        if self.b is not None: # bias
            y = self.add.fwd(y, self.b.mat)  # broadcast
        y = y.reshape(B, self.oh, self.ow, self.out_dim).transpose(0, 3, 1, 2)  # reshape back to [N, out_dim, out_h, out_w]
        return y

    def bwd(self, dl):
        B, _, out_h, out_w = dl.shape # dl: [N, C, out_h, out_w]
        dl = dl.transpose(0, 2, 3, 1).reshape(-1, self.out_dim) # reshape dl to [N*out_h*out_w, out_dim]
        if self.b is not None:
            dl, db = self.add.bwd(dl)
        dx, dw = self.mul.bwd(dl)
        dw = dw.reshape(self.w.mat.shape)
        if self.req_grad:
            if self.w.grad is None:
                self.w.grad = dw
            else:
                self.w.grad += dw
            if self.b is not None:
                if self.b.grad is None:
                    self.b.grad = db
                else:
                    self.b.grad += db
        dx = col2im(dx, self.xp_shape, self.oh, self.ow, self.fh, self.fw, self.stride)  # dx reshape using col2im, restores to x_pad_shape
        if self.pad > 0:
            dx = dx[:, :, self.pad:-self.pad, self.pad:-self.pad] # back to pre-padding
        return dx


# Transform
class Transpose(Mod):
    def __init__(self, *axis):
        super().__init__()
        self.new = axis # tuple
        self.old = [0] * len(self.new)
        for i, a in enumerate(self.new):  # index inversion
            self.old[a] = i

    def fwd(self, x):
        return x.transpose(self.new)

    def bwd(self, dl):
        return dl.transpose(self.old)


class Reshape(Mod):
    def __init__(self, *shape):
        super().__init__()
        self.new = list(shape) # tuple
        self.old = None

    def fwd(self, x):
        self.old = x.shape
        #return x.reshape(*self.new)
        tmp = self.new.copy()
        if -1 in tmp:
            idx = tmp.index(-1)
            neg = 1
            for d in tmp:
                if d != -1:
                    neg *= d
            tmp[idx] = int(np.prod(self.old)/neg)
        return x.reshape(tmp)

    def bwd(self, dl):
        return dl.reshape(self.old)


# optimizer
class SGD:
    def __init__(self, params, lr=1e-3):
        self.params = params #-> list
        self.lr = lr
    def step(self):
        for p in self.params:
            if p.grad is not None:
                p.mat -= self.lr * p.grad
    def zero_grad(self): # cleanup
        for p in self.params:
            p.grad = None


class SGD_M: # with momentum
    def __init__(self, params, lr=1e-3, momentum=0.9):
        self.params = params
        self.lr = lr
        self.m = momentum
        self.v = [np.zeros_like(p.mat) for p in params] # vel
    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is not None:
                self.v[i] = self.m * self.v[i] - self.lr * p.grad
                p.mat += self.v[i]
    def zero_grad(self):
        for p in self.params:
            p.grad = None


class RMSprop:
    def __init__(self, params, lr=1e-3, beta=0.9, eps=1e-8):
        self.params = params
        self.lr = lr
        self.b = beta
        self.eps = eps
        self.v = [np.zeros_like(p.mat) for p in params]
    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is not None:
                self.v[i] = self.b*self.v[i] + (1-self.b)*(p.grad*p.grad)
                p.mat -= self.lr * p.grad/(math.sqrt(self.v[i])+self.eps)
    def zero_grad(self):
        for p in self.params:
            p.grad = None


class Adam:
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = params
        self.lr = lr
        self.b1 = beta1
        self.b2 = beta2
        self.eps = eps
        self.m = [np.zeros_like(p.mat) for p in params]
        self.v = [np.zeros_like(p.mat) for p in params]
        self.t = 0
    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is not None:
                self.m[i] = self.b1*self.m[i] + (1-self.b1)*p.grad
                self.v[i] = self.b2*self.v[i] + (1-self.b2)*(p.grad*p.grad)
                m_hat = self.m[i]/(1-self.b1**self.t) # b correction
                v_hat = self.v[i]/(1-self.b2**self.t)
                p.mat -= self.lr * m_hat/(np.sqrt(v_hat)+self.eps)
    def zero_grad(self):
        for p in self.params:
            p.grad = None


class Adam_W:
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, weight_decay=0.01, eps=1e-8):
        self.params = params
        self.lr = lr
        self.wd = weight_decay
        self.b1 = beta1
        self.b2 = beta2
        self.eps = eps
        self.m = [np.zeros_like(p.mat) for p in params]
        self.v = [np.zeros_like(p.mat) for p in params]
        self.t = 0
    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is not None:
                self.m[i] = self.b1*self.m[i] + (1-self.b1)*p.grad
                self.v[i] = self.b2*self.v[i] + (1-self.b2)*(p.grad*p.grad)
                m_hat = self.m[i]/(1-self.b1**self.t) # b correction
                v_hat = self.v[i]/(1-self.b2**self.t)
                p.mat -= self.lr * m_hat/(np.sqrt(v_hat)+self.eps)
                p.mat -= self.lr * self.wd * p.mat # w decay
    def zero_grad(self):
        for p in self.params:
            p.grad = None


# Model def
class ModList(Mod):
    def __init__(self, *layer):
        super().__init__()
        self.list = []
        for i, j in enumerate(layer):
            self.list.append(j)
            setattr(self, str(i), j) # self.i = layer

    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):
        return self.list[idx]

    def __iter__(self):
        return iter(self.list)

    def append(self, layer):
        setattr(self, str(len(self.list)), layer)
        self.list.append(layer)
        return


def save_ckpt(model, optim, epoch, path="./model.npydl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {}
    for i, p in enumerate(model.params()):
        data[f"param_{i}"] = p.mat

    for i, (m, v) in enumerate(zip(optim.m, optim.v)):
        data[f"m_{i}"] = m
        data[f"v_{i}"] = v

    data["epoch"] = epoch
    np.savez(open(path, "wb"), **data)


def load_ckpt(model, optim, path="./model.npydl"):
    if os.path.exists(path):
        data = np.load(path)
        for i, p in enumerate(model.params()):
            p.mat[:] = data[f"param_{i}"]

        for i in range(len(optim.m)):
            optim.m[i][:] = data[f"m_{i}"]
            optim.v[i][:] = data[f"v_{i}"]

        epoch = int(data["epoch"])
        return epoch
    else:
        print(f"{path} not found")
        return 0


