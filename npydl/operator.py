import numpy as np
import math


# operator
class Matmul:
    def __init__(self):
        self.x = None
        self.w = None
    def fwd(self, x, w):
        self.x = x
        self.w = w
        #return np.dot(x, w) # y = xw
        return np.matmul(x, w) # y = xw
    def bwd(self, dl):
        #dx = np.dot(dl, self.w.T) # dL/dx = dL/dy * dy/dx
        #dx = np.matmul(dl, self.w.T) # dL/dx = dL/dy * dy/dx
        #dw = np.dot(self.x.T, dl)
        #dw = np.matmul(self.x.reshape(-1, self.x.shape[-1]).T, dl.reshape(-1, dl.shape[-1]))

        wT = np.swapaxes(self.w, -1, -2)
        xT = np.swapaxes(self.x, -1, -2)
        dx = np.matmul(dl, wT)
        dw = np.matmul(xT, dl)
        if self.w.ndim == 2:
            axes = tuple(range(dw.ndim - 2))
            if axes:
                dw = np.sum(dw, axis=axes)

        return dx, dw


class Matadd:
    def __init__(self):
        self.x_shape = None
        self.y_shape = None
    def fwd(self, x, y):
        self.x_shape = x.shape
        self.y_shape = y.shape
        return x + y
    def shapematch(self, grad, tgt_shape): # shape match for broadcast gradient
        while grad.ndim > len(tgt_shape): # delete
            grad = np.sum(grad, axis=0, keepdims=False)
        for axis, size in enumerate(tgt_shape): # restore to tgt_shape (for the cases like [1, b])
            if size == 1:
                grad = np.sum(grad, axis=axis, keepdims=True)
        return grad.reshape(tgt_shape)
    def bwd(self, dl):
        dx = dl.copy() # dl * 1
        dy = dl.copy() # dl * 1
        if dx.shape != self.x_shape: # for broadcasting
            #dx = np.sum(dx, axis=0, keepdims=True)
            dx = self.shapematch(dx, self.x_shape)
        if dy.shape != self.y_shape:
            #dy = np.sum(dy, axis=0, keepdims=True)
            dy = self.shapematch(dy, self.y_shape)
        return dx, dy