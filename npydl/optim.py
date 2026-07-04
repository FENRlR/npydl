import numpy as np
import math


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
