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
    if True:
        # Try out the generator model with random input
        latent_space_samples = torch.randn(1, 1, 28, 28, device=GPU_DEVICE)
    else:
        # see what the model does with real input
        latent_space_samples = next(
            iter(
                torch.utils.data.DataLoader(
                    GAN.data(), batch_size=1, shuffle=True
                )
            )
        )[0]
        print(latent_space_samples)
        print(latent_space_samples.size())
        if GPU_DEVICE:
            latent_space_samples = latent_space_samples.cuda()
    # load the model
    gan = GAN.load("GAN_4198.6636476")
    # gen output
    generated_samples = gan.generator(latent_space_samples)
    generated_samples = generated_samples.detach()
    # Try out the discriminator model:
    decision = gan.discriminator(generated_samples)
    decision = decision.detach()
    if GPU_DEVICE:
        generated_samples = generated_samples.cpu()
        latent_space_samples = latent_space_samples.cpu()
        decision = decision.cpu()
    # INPUT
    print("INPUT:")
    print("--------------")
    plt.imshow(latent_space_samples[0, 0, :, :])
    plt.figure()
    # OUTPUT
    print(generated_samples)
    print(generated_samples.size())
    print("--------------")
    print(decision)
    plt.imshow(generated_samples[0, 0, :, :])
    plt.show()

if __name__ == "__main__":
    main()
    