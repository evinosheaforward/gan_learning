import math
import os
import timeit

import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms

from trainer import GAN

# Use GPU switch (TODO: make this an arg ofc)
GPU_DEVICE = torch.device("cuda")  # Default CUDA device
# GPU_DEVICE = None


def main():
    # load the model
    latent_space_samples = GAN.latent_input()
    files = sorted(
        [
            i[:-3] for i in os.listdir("./models")
            if i.startswith("GAN_new") and i.endswith("gen")
            # and not "lower_lr" in i
            and "lower_lr" in i
        ],
        key=lambda x: float(x.split("_")[-1])
    )
    i, j = 1, 0
    xlen = math.floor(math.sqrt(len(files)))
    ylen = math.ceil(len(files) / xlen ) - 1
    xlen += 1

    _, axes = plt.subplots(xlen, ylen)
    for file in files:
        print(file)
        gan = GAN.load("models/" + file)
        # gen output
        generated_samples = gan.generator(latent_space_samples)
        generated_samples = generated_samples.detach()
        # Try out the discriminator model:
        decision = gan.discriminator(generated_samples)
        decision = decision.detach()
        if GPU_DEVICE:
            generated_samples = generated_samples.cpu()
            decision = decision.cpu()
        # OUTPUT
        print(decision)
        axes[j, i].imshow(generated_samples[0, 0, :, :])
        if i == xlen-1:
            i = 0
            j += 1
        else:
            i += 1
    latent_space_samples = latent_space_samples.cpu()
    axes[0, 0].imshow(latent_space_samples[0, 0, :, :])
    plt.show()

def test_model():
    # load the model
    file = "GAN_new_lower_lr_20819.5471887"
    xlen = 12
    ylen = 12
    gan = GAN.load("models/" + file)
    print(file)

    _, axes = plt.subplots(xlen, ylen)
    for i in range(xlen):
        for j in range(ylen):
            latent_space_samples = GAN.latent_input()
            # gen output
            generated_samples = gan.generator(latent_space_samples)
            generated_samples = generated_samples.detach()
            # Try out the discriminator model:
            decision = gan.discriminator(generated_samples)
            decision = decision.detach()
            if GPU_DEVICE:
                generated_samples = generated_samples.cpu()
                decision = decision.cpu()
            # OUTPUT
            print(decision)
            axes[j, i].imshow(generated_samples[0, 0, :, :])
    # latent_space_samples = latent_space_samples.cpu()
    # axes[0, 0].imshow(latent_space_samples[0, 0, :, :])
    plt.show()


if __name__ == "__main__":
    test_model()
    # main()
    