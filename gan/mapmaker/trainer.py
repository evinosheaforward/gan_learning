import argparse
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
    plt.savefig(f"outputs/losses_mapmaker_{timeit.default_timer()}")
    plt.clf()


def main(model_path=None):
    # to train saved or load new model
    if model_path:
        gan = GAN.load(model_path, mode="train")
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
        plot_losses(disc_losses, gen_losses)
        gan.generate_image()


if __name__ == "__main__":
    print(torch.cuda.is_available())
    print(cuda.detect())
    print(torch.cuda.get_device_name(torch.cuda.current_device()))
    parser = argparse.ArgumentParser(description='Train the mapmaker GAN.')
    parser.add_argument(
        '--load',
        dest='model_path',
        type=str, 
        default=None,
        help='path to model to load, will start from scratch if not specified',
    )
    args = parser.parse_args()
    main(model_path=args.model_path)

"""
TODO: 
X try out numba for using graphics card
X try running algorithm with numba
- try working on a new GAN
"""
