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
from torchvision import datasets, transforms

from gan.mapmaker import mapimg

# Use GPU switch (TODO: make this an arg ofc)
GPU_DEVICE = torch.device("cuda")  # Default CUDA device


def add_noise(in_tensor, percent=0.15):
    return (
        torch.rand(in_tensor.size(), device=GPU_DEVICE) * percent
        + (1.0 - percent) * in_tensor
    )


class Reshape(torch.nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(x.shape[0], *self.shape)


class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.reshape(x.shape[0], -1)


class Generator(nn.Module):
    def __init__(self):
        self.in_ch, self.in_x, self.in_y = 8, 8, 8
        super().__init__()

        self.model = nn.Sequential(
            Flatten(),
            nn.Linear(self.in_ch * self.in_x * self.in_y, 256 * 8 * 8),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            Reshape((256, 8, 8)),
            nn.Dropout(0.3),
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=128,
                kernel_size=(6, 6),
                stride=(2, 2),
                padding=(2, 2),
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=(6, 6),
                stride=(2, 2),
                padding=(2, 2),
            ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=32,
                kernel_size=(6, 6),
                stride=(2, 2),
                padding=(2, 2),
            ),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=3,
                kernel_size=(6, 6),
                stride=(2, 2),
                padding=(2, 2),
            ),
            nn.Tanh(),
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
                in_channels=3,
                out_channels=64,
                kernel_size=(6, 6),
                stride=(2, 2),
                padding=(2, 2),
            ),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(6, 6),
                stride=(2, 2),
                padding=(2, 2),
            ),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(6, 6),
                stride=(2, 2),
                padding=(2, 2),
            ),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(6, 6),
                stride=(2, 2),
                padding=(2, 2),
            ),
            nn.LeakyReLU(0.2),
            Flatten(),
            nn.Dropout(0.4),
            nn.Linear(4096, 2048),
            nn.Linear(2048, 1),
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


class MapMaker:
    def __init__(self, discriminator=Discriminator(), generator=Generator()):
        self.discriminator = discriminator
        self.generator = generator

    @classmethod
    def load(cls, path, mode="eval"):
        return cls(
            Discriminator.load(path + "desc", mode=mode),
            Generator.load(path + "gen", mode=mode),
        )

    def save(self, path):
        self.discriminator.save(path + "desc")
        self.generator.save(path + "gen")

    def load_train_data(self, path="data/dnd_maps/"):
        self.train_data = mapimg.load_images_torchvision("data/dnd_maps/")

    def shuffle_data(self, batch_size):
        return torch.utils.data.DataLoader(
            self.train_data, batch_size=batch_size, shuffle=True
        )

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

    def train(self, noise=False):
        """Train the model by iterating through the dataset
        num_epoch times, printing the duration per epoch
        """
        batch_size = 1
        num_epochs = 10
        # Labels for real data:
        # - for discriminator, this is real images
        # - for generator this is what we wanted the discriminator output to be
        real_samples_labels = torch.full((batch_size, 1), 0.9, device=GPU_DEVICE)
        # Init loss functions
        loss_function = nn.BCELoss()
        gen_losses = []
        disc_losses = []
        # total data is dataset * num_epochs
        # Load train data
        train_loader = self.shuffle_data(batch_size)
        self.generator.model.eval()
        self.discriminator.model.train()
        # Labels for generated data, all 0
        generated_samples_labels = torch.zeros((batch_size, 1), device=GPU_DEVICE)
        # Load optimizer
        optimizer_discriminator = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=0.0004,
        )
        # # self.generator.model.train()
        # # self.discriminator.model.eval()
        # total data is batch_size * num_epochs
        # Load optimizer
        optimizer_generator = torch.optim.Adam(
            self.generator.parameters(),
            lr=0.0001,
        )
        start = timeit.default_timer()
        # Repeat num_epoch times
        for epoch in range(num_epochs):
            for n, images in enumerate(train_loader):
                # Iterate through dataset
                if GPU_DEVICE:
                    images = images.cuda()
                # Data for training the discriminator
                latent_space_samples = self.latent_input(batch_size)
                generated_samples = self.generator(latent_space_samples)
                # label inputs as real, fake
                all_samples = torch.cat((images, generated_samples))
                if noise:
                    all_samples_labels = add_noise(
                        torch.cat((real_samples_labels, generated_samples_labels)),
                        percent=noise,
                    )
                else:
                    all_samples_labels = torch.cat(
                        (real_samples_labels * 2.0 - 1.0, generated_samples_labels)
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
                if noise:
                    generated_samples = self.generator(latent_space_samples)
                else:
                    generated_samples = self.generator(latent_space_samples)
                output_discriminator_generated = self.discriminator(generated_samples)
                loss_generator = loss_function(
                    output_discriminator_generated, real_samples_labels
                )
                loss_generator.backward()
                optimizer_generator.step()
                gen_losses.append(float(loss_generator))

            if epoch % 10 == 0:
                # Show loss
                print(f"Epoch: {epoch} Loss D.: {loss_discriminator}")
                print(f"Epoch: {epoch} Loss G.: {loss_generator}")
                print(timeit.default_timer() - start)
                start = timeit.default_timer()
        return disc_losses, gen_losses

    def latent_input(self, batch_size=1, generated=True):
        return torch.randn(
            batch_size,
            self.generator.in_ch,
            self.generator.in_x,
            self.generator.in_y,
            device=GPU_DEVICE,
        )

    def generate_image(self, return_it=False, save=True, output_dir="outputs/"):
        if save:
            output = self.generator(self.latent_input())
            plt.imsave(
                output_dir
                + f"mapmaker_batchnorm_{timeit.default_timer()}".replace(".", "_")
                + ".jpg",
                # Using tanh activation function, but rbg is 0..1, so do (X+1)/2.0
                (output[0, :, :, :].detach().cpu().permute(1, 2, 0).numpy() + 1.0)
                / 2.0,
            )
        elif return_it:
            return self.generator(self.latent_input(batch_size=return_it))
        else:
            output = self.generator(self.latent_input())
            plt.imshow(
                (output[0, :, :, :].detach().cpu().permute(1, 2, 0).numpy() + 1.0)
                / 2.0,
            )
            plt.show()

    @staticmethod
    def discriminator_latent_input(batch_size=1, generated=True):
        return torch.randn(batch_size, 3, 128, 128, device=GPU_DEVICE)
