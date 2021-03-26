from functools import partial, reduce
import math
from operator import __add__
import timeit

import matplotlib.pyplot as plt
import numpy
from numba import jit, cuda
import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms

from models import GAN

# Use GPU switch (TODO: make this an arg ofc)
GPU_DEVICE = torch.device("cuda")  # Default CUDA device


def unzip(indata):
    x, y = [], []
    for i, j in indata:
        x.append(i)
        y.append(j)
    return x, y


def plot_losses(disc_loss, gen_loss):
    for i, losses in enumerate(disc_loss):
        plt.plot(*unzip(enumerate(disc_loss)), label=f"disc: {i}")
    for i, losses in enumerate(gen_loss):
        plt.plot(*unzip(enumerate(losses)), label=f"gen: {i}")
    plt.legend()
    plt.show()


def main():
    # to train saved or load new model
    if True:
        gan = GAN.load("models/mapmaker/mapmaker_batchnorm_377.4695443", mode="train")
    else:
        gan = GAN()

    gan.load_train_data()
    disc_losses, gen_losses = [], []
    for i in range(60):
        # Train the models
        start = timeit.default_timer()
        dl, gl = gan.train()
        disc_losses.append(dl)
        gen_losses.append(gl)
        print("Train Time:")
        print(timeit.default_timer() - start)
        gan.save(f"models/mapmaker_batchnorm_{timeit.default_timer()}")
        # plot_losses(disc_losses, gen_losses)
        output = gan.generator(gan.latent_input()).cpu()
        plt.imsave(
            f"outputs/mapmaker_batchnorm_{timeit.default_timer()}".replace(".", "_")
            + ".jpg",
            output[0, :, :, :].detach().permute(1, 2, 0).numpy(),
        )


if __name__ == "__main__":
    print(torch.cuda.is_available())
    print(cuda.detect())
    print(torch.cuda.get_device_name(torch.cuda.current_device()))
    main()

"""
TODO: 
X try out numba for using graphics card
X try running algorithm with numba
- try working on a new GAN
"""
