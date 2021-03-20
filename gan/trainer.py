from functools import partial
import math
import timeit

import matplotlib.pyplot as plt
import numpy
from numba import jit, cuda
import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms

GPU_DEVICE = torch.device("cuda")  # Default CUDA device
# GPU_DEVICE = None


def plot(train_data):
    plt.plot(train_data[:, 0], train_data[:, 1], ".")

class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3),
            nn.Conv2d(
                in_channels=4, out_channels=1, kernel_size=5,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Flatten(),
            nn.Linear(25, 1),
            nn.Sigmoid(),
        )
        if GPU_DEVICE:
            self.model = self.model.cuda()

    def forward(self, x):
        return self.model(x)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=12, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Upsample(scale_factor=4),
            nn.Conv2d(
                in_channels=12, out_channels=6, kernel_size=5, stride=2, padding=1
            ),
            nn.Conv2d(
                in_channels=6, out_channels=1, kernel_size=2, stride=1, padding=1
            ),
            nn.ReLU(),
        )
        if GPU_DEVICE:
            self.model = self.model.cuda()

    def forward(self, x):
        return self.model(x)


class GAN:
    def __init__(self, discriminator, generator):
        self.discriminator = discriminator
        self.generator = generator

    @staticmethod
    def data():
        return torchvision.datasets.FashionMNIST(
            root="./data/FashionMNIST",
            train=True,
            download=True,
            transform=transforms.Compose([transforms.ToTensor()]),
        )

    def train(self):
        lr = 0.001
        num_epochs = 300
        batch_size = 32
        loss_function = nn.BCELoss()

        train_set = self.data()
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=True
        )
        optimizer_discriminator = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr
        )
        optimizer_generator = torch.optim.Adam(self.generator.parameters(), lr=lr)

        for epoch in range(num_epochs):
            start = timeit.default_timer()
            # for n, (real_samples, _) in enumerate(train_loader):
            for n, (images, labels) in enumerate(train_loader):
                if GPU_DEVICE:
                    images = images.cuda()
                # Data for training the discriminator
                real_samples_labels = torch.ones(
                    (batch_size, 1), device=GPU_DEVICE
                )
                latent_space_samples = torch.randn(
                    (batch_size, 1, 28, 28), device=GPU_DEVICE
                )
                generated_samples = self.generator(latent_space_samples)
                generated_samples_labels = torch.zeros(
                    (batch_size, 1), device=GPU_DEVICE
                )
                all_samples = torch.cat((images, generated_samples))
                all_samples_labels = torch.cat(
                    (real_samples_labels, generated_samples_labels)
                )

                # Training the discriminator
                self.discriminator.zero_grad()
                output_discriminator = self.discriminator(all_samples)
                loss_discriminator = loss_function(
                    output_discriminator, all_samples_labels
                )
                loss_discriminator.backward()
                optimizer_discriminator.step()

                # Data for training the generator
                latent_space_samples = torch.randn(
                    (batch_size, 1, 28, 28), device=GPU_DEVICE
                )

                # Training the generator
                self.generator.zero_grad()
                generated_samples = self.generator(latent_space_samples)
                output_discriminator_generated = self.discriminator(generated_samples)
                loss_generator = loss_function(
                    output_discriminator_generated, real_samples_labels
                )
                loss_generator.backward()
                optimizer_generator.step()

                # Show loss
                if epoch % 10 == 0 and n == batch_size - 1:
                    print(f"Epoch: {epoch} Loss D.: {loss_discriminator}")
                    print(f"Epoch: {epoch} Loss G.: {loss_generator}")
                    print(timeit.default_timer() - start)


def main():
    gan = GAN(Discriminator(), Generator())
    gan.train()
    latent_space_samples = torch.randn(1, 1, 28, 28, device=GPU_DEVICE)
    generated_samples = gan.generator(latent_space_samples)
    generated_samples = generated_samples.detach()
    if GPU_DEVICE:
        generated_samples = generated_samples.cpu()
    print(generated_samples)
    print(generated_samples.size())

    # generated_samples = gan.discriminator(latent_space_samples)
    # generated_samples = generated_samples.detach()
    # if GPU_DEVICE:
    #     generated_samples = generated_samples.cpu()
    # print(generated_samples)
    # print(generated_samples.size())

    plt.imshow(generated_samples[0, 0, :, :])
    plt.show()


if __name__ == "__main__":
    print(torch.cuda.is_available())
    print(cuda.detect())
    print(torch.cuda.get_device_name(torch.cuda.current_device()))
    start = timeit.default_timer()
    main()
    # print(data())
    print(timeit.default_timer() - start)

"""
TODO: 
X try out numba for using graphics card
X try running algorithm with numba
- try working on a new GAN
"""
