import numpy as np
import os


# save checkpoint
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