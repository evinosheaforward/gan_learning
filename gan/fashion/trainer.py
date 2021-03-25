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

# Use GPU switch (TODO: make this an arg ofc)
GPU_DEVICE = torch.device("cuda")  # Default CUDA device
# GPU_DEVICE = None


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


class Reshape(torch.nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.shape[0], *self.shape)


class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            Flatten(),
            nn.Linear(28 * 28, 128 * 7 * 7),
            nn.LeakyReLU(0.2),
            Reshape((128, 7, 7)),
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=128,
                kernel_size=(4, 4),
                stride=(2, 2),
                padding=(1, 1),
            ),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=128,
                kernel_size=(4, 4),
                stride=(2, 2),
                padding=(1, 1),
            ),
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                in_channels=128, out_channels=1, kernel_size=(7, 7), padding=(3, 3)
            ),
        )
        if GPU_DEVICE:
            self.model = self.model.cuda()

    def forward(self, x):
        return self.model(x)

    @classmethod
    def load(cls, path, mode="eval"):
        inst = cls()
        inst.model.load_state_dict(torch.load(path), strict=False)
        if mode == "eval":
            inst.model.eval()
        else:
            inst.model.train()
        return inst

    def save(self, path):
        torch.save(self.model.state_dict(), path)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=(4, 4),
                stride=(2, 2),
                padding=(1, 1),
            ),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(4, 4),
                stride=(2, 2),
                padding=(1, 1),
            ),
            nn.LeakyReLU(0.2),
            Flatten(),
            nn.Linear(3136, 1),
            nn.Sigmoid(),
        )
        if GPU_DEVICE:
            self.model = self.model.cuda()

    def forward(self, x):
        return self.model(x)

    @classmethod
    def load(cls, path, mode="eval"):
        inst = cls()
        inst.model.load_state_dict(torch.load(path), strict=False)
        if mode == "eval":
            inst.model.eval()
        else:
            inst.model.train()
        return inst

    def save(self, path):
        torch.save(self.model.state_dict(), path)


class GAN:
    def __init__(self, discriminator, generator):
        self.discriminator = discriminator
        self.generator = generator

    def performance(self, step, n_samples=5):
        # prepare fake examples
        generated_samples = gan.generator(GAN.latent_input(n_samples))
        generated_samples = generated_samples.detach()
        if GPU_DEVICE:
            generated_samples = generated_samples.cpu()
        # plot images
        for i in range(10 * 10):
            # define subplot
            plt.subplot(10, 10, 1 + i)
            # turn off axis
            plt.axis("off")
            # plot raw pixel data
            plt.imshow(X[i, 0, :, :], cmap="gray_r")
        # save plot to file
        plt.savefig("results/generator_step_%03d.png" % (step + 1))
        plt.close()

    @staticmethod
    def data():
        return torchvision.datasets.FashionMNIST(
            root="./data/FashionMNIST",
            train=True,
            download=True,
            transform=transforms.Compose([transforms.ToTensor()]),
        )

    @classmethod
    def load(cls, path, mode="eval"):
        return cls(
            Discriminator.load(path + "desc", mode=mode),
            Generator.load(path + "gen", mode=mode),
        )

    def save(self, path):
        self.discriminator.save(path + "desc")
        self.generator.save(path + "gen")

    def train(self):
        """Train the model by iterating through the dataset
        num_epoch times, printing the duration per epoch
        """
        lr = 0.0002
        batch_size = 1000
        num_epochs = 10
        # Labels for real data:
        # - for discriminator, this is real images
        # - for generator this is what we wanted the discriminator output to be
        real_samples_labels = torch.ones((batch_size, 1), device=GPU_DEVICE)
        # Init loss functions
        loss_function = nn.BCELoss()
        gen_losses = []
        disc_losses = []
        # total data is dataset * num_epochs
        # Load train data
        train_set = self.data()
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=True
        )
        self.generator.model.eval()
        self.discriminator.model.train()
        # Labels for generated data, all 0
        generated_samples_labels = torch.zeros((batch_size, 1), device=GPU_DEVICE)
        # Load optimizer
        optimizer_discriminator = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=lr,
        )
        self.generator.model.train()
        self.discriminator.model.eval()
        # total data is batch_size * num_epochs
        # Load optimizer
        optimizer_generator = torch.optim.Adam(
            self.generator.parameters(),
            lr=lr,
        )
        start = timeit.default_timer()
        # Repeat num_epoch times
        for epoch in range(num_epochs):
            for n, (images, labels) in enumerate(train_loader):
                # Iterate through dataset
                if GPU_DEVICE:
                    images = images.cuda()
                # Data for training the discriminator
                latent_space_samples = self.latent_input(batch_size)
                generated_samples = self.generator(latent_space_samples)
                # label inputs as real, fake
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
                disc_losses.append(float(loss_discriminator))
                # Data for training the generator
                latent_space_samples = self.latent_input(batch_size)
                # Training the generator
                self.generator.zero_grad()
                generated_samples = self.generator(latent_space_samples)
                output_discriminator_generated = self.discriminator(generated_samples)
                loss_generator = loss_function(
                    output_discriminator_generated, real_samples_labels
                )
                loss_generator.backward()
                optimizer_generator.step()
                gen_losses.append(float(loss_generator))
            if epoch % (x := 10) == 0:
                # Show loss
                print(f"Epoch: {epoch} Loss D.: {loss_discriminator}")
                print(f"Epoch: {epoch} Loss G.: {loss_generator}")
                print(timeit.default_timer() - start)
                start = timeit.default_timer()
        return disc_losses, gen_losses

    @staticmethod
    def latent_input(batch_size=1, generated=True):
        return torch.randn(batch_size, 1, 28, 28, device=GPU_DEVICE)


def main():
    # to train saved or load new model
    if False:
        gan = GAN.load("models/GAN_new_1182.0918666", mode="train")
    else:
        gan = GAN(Discriminator(), Generator())

    disc_losses, gen_losses = [], []
    for i in range(60):
        # Train the models
        start = timeit.default_timer()
        dl, gl = gan.train()
        disc_losses.append(dl)
        gen_losses.append(gl)
        print("Train Time:")
        print(timeit.default_timer() - start)
        gan.save(f"models/GAN_new_lower_lr_{timeit.default_timer()}")
        # plot_losses(disc_losses, gen_losses)


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
