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

from gan.mapmaker.models import MapMaker

# Use GPU switch (TODO: make this an arg ofc)
GPU_DEVICE = torch.device("cuda")  # Default CUDA device


def unzip(indata):
    x, y = [], []
    for i, j in indata:
        x.append(i)
        y.append(j)
    return x, y


def plot_losses(disc_loss, gen_loss):
    plt.plot(*unzip(enumerate(disc_loss)), label=f"discriminator")
    plt.plot(*unzip(enumerate(gen_loss)), label=f"generator")
    plt.legend()
    plt.savefig(
        f"outputs/losses_mapmaker_{timeit.default_timer()}".replace(".", "_") + ".png"
    )
    plt.clf()


def main(model_path=None, noise=False):
    # to train saved or load new model
    if model_path:
        gan = MapMaker.load(model_path, mode="train")
    else:
        gan = MapMaker()

    gan.load_train_data()
    print("finished loading training data")
    disc_losses, gen_losses = [], []
    for i in range(60):
        # Train the models
        start = timeit.default_timer()
        dl, gl = gan.train(noise=noise)
        disc_losses.extend(dl)
        gen_losses.extend(gl)
        print("Train Time:")
        print(timeit.default_timer() - start)
        gan.save(f"models/mapmaker_batchnorm_{timeit.default_timer()}")
        plot_losses(disc_losses, gen_losses)
        gan.generate_image()


if __name__ == "__main__":
    print(torch.cuda.is_available())
    print(cuda.detect())
    print(torch.cuda.get_device_name(torch.cuda.current_device()))
    parser = argparse.ArgumentParser(description="Train the mapmaker GAN.")
    parser.add_argument(
        "--load",
        dest="model_path",
        type=str,
        default=None,
        help="path to model to load, will start from scratch if not specified",
    )
    parser.add_argument(
        "--noise",
        dest="noise",
        default=False,
        type=int,
        action="store",
        help="whether or not to add noise to the imags before passing to discriminator during training",
    )
    args = parser.parse_args()
    print("START")
    main(model_path=args.model_path, noise=args.noise / 100.0)
