import os
import shutil

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy

from lxml import html
import requests

from skimage.transform import resize

from torchvision import datasets, transforms
import torch

# Use GPU switch (TODO: make this an arg ofc)
GPU_DEVICE = torch.device("cuda")  # Default CUDA device
# GPU_DEVICE = None


def load_images(path, num=-1):
    train_set = []
    count = 0
    for fname in os.listdir(path):
        fpath = os.path.join(path, fname)
        img = mpimg.imread(fpath).astype(numpy.uint8)
        # plt.imshow(img)
        # plt.figure()
        train_set.append(resize(img, (512, 512, 3)))
        count += 1
        if count == num:
            break
    return (
        torch.from_numpy(
            numpy.asarray(train_set)
            # given the fact that the activation is tanh,
            # the output of the generator is applied with f(x): (x+1)/2
            # to get RGB images (0..1)
            # so, applying f-inverse(x): (x*2)-1
            # to the training data to get training data domain (-1..1)
        ).permute(0, 3, 1, 2)
        * 2
        - 1
    ).float()


def load_images_torchvision(path="data/temp_maps/"):
    transform = transforms.Compose(
        [
            transforms.Resize(1024),
            transforms.ToTensor(),
        ]
    )
    print(len(os.listdir("data/dnd_maps")))
    return datasets.ImageFolder(root=path, transform=transform)
