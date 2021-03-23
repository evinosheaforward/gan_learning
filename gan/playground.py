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
    _, axes = plt.subplots(10, 10)
    i, j = 0, 0
    for file in sorted(
        [
            i[:-3] for i in os.listdir("./models")
            if i.startswith("GAN_new") and i.endswith("gen")
            # and not "lower_lr" in i
            # and "lower_lr" in i
        ],
        key=lambda x: float(x.split("_")[-1])
    ):
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
        axes[i, j].imshow(generated_samples[0, 0, :, :])
        if i == 10:
            i = 0
            j += 1
        else:
            i += 1
    latent_space_samples = latent_space_samples.cpu()
    axes[i, j].imshow(latent_space_samples[0, 0, :, :])
    plt.show()

if __name__ == "__main__":
    main()
    