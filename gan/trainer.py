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


def data():
    # torch.manual_seed(111)
    train_data_length = 1024
    train_data = torch.zeros((train_data_length, 2), device=GPU_DEVICE)
    train_data[:, 0] = 2 * math.pi * torch.rand(train_data_length, device=GPU_DEVICE)
    train_data[:, 1] = torch.sin(train_data[:, 0])
    train_labels = torch.zeros(train_data_length, device=GPU_DEVICE)
    # train_set = [
    #     (train_data[i], train_labels[i]) for i in range(train_data_length)
    # ]

    train_set = torchvision.datasets.FashionMNIST(
        root="./data/FashionMNIST",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    return train_data, train_set, train_labels


def plot(train_data):
    plt.plot(train_data[:, 0], train_data[:, 1], ".")


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
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
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )
        if GPU_DEVICE:
            self.model = self.model.cuda()

    def forward(self, x):
        return self.model(x)


def train(discriminator, generator):
    lr = 0.001
    num_epochs = 300
    batch_size = 32

    loss_function = nn.BCELoss()
    train_data, train_set, train_labels = data()
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True
    )
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)

    for epoch in range(num_epochs):
        start = timeit.default_timer()
        # for n, (real_samples, _) in enumerate(train_loader):
        for n, (images, labels) in enumerate(train_loader):
            print(images.size())
            return
            if GPU_DEVICE:
                images.cuda()
            # Data for training the discriminator
            real_samples_labels = torch.ones((batch_size, 1, 28, 28), device=GPU_DEVICE)
            latent_space_samples = torch.randn(
                (batch_size, 2, 28, 28), device=GPU_DEVICE
            )
            generated_samples = generator(latent_space_samples)
            generated_samples_labels = torch.zeros((batch_size, 1), device=GPU_DEVICE)
            all_samples = torch.cat((real_samples, generated_samples))
            all_samples_labels = torch.cat(
                (real_samples_labels, generated_samples_labels)
            )

            # Training the discriminator
            discriminator.zero_grad()
            output_discriminator = discriminator(all_samples)
            loss_discriminator = loss_function(output_discriminator, all_samples_labels)
            loss_discriminator.backward()
            optimizer_discriminator.step()

            # Data for training the generator
            latent_space_samples = torch.randn((batch_size, 2), device=GPU_DEVICE)

            # Training the generator
            generator.zero_grad()
            generated_samples = generator(latent_space_samples)
            output_discriminator_generated = discriminator(generated_samples)
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
    discriminator = Discriminator()
    generator = Generator()
    train(discriminator, generator)
    latent_space_samples = torch.randn(100, 2, device=GPU_DEVICE)
    generated_samples = generator(latent_space_samples)
    generated_samples = generated_samples.detach()
    # plt.plot(generated_samples[:, 0].cpu(), generated_samples[:, 1].cpu(), ".")
    # plt.show()


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
