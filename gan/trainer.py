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

# Use GPU switch (TODO: make this an arg ofc)
GPU_DEVICE = torch.device("cuda")  # Default CUDA device
# GPU_DEVICE = None


def plot(train_data):
    plt.plot(train_data[:, 0], train_data[:, 1], ".")

class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=6,
                kernel_size=24,
                stride=3,
                padding=4
            ),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(0.25),
            nn.Conv2d(
                in_channels=6,
                out_channels=4,
                kernel_size=24,
                stride=1,
                padding=2
            ),
            nn.Dropout2d(0.25),
            nn.Conv2d(
                in_channels=4,
                out_channels=1,
                kernel_size=10,
                stride=2,
                padding=2
            ),
            nn.Conv2d(
                in_channels=1,
                out_channels=1,
                kernel_size=7,
                stride=1,
                padding=1
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
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3),
            nn.Conv2d(
                in_channels=4, out_channels=1, kernel_size=5,
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            Flatten(),
            nn.Linear(25, 1),
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

    def train(self, model="generator"):
        """Train the model by iterating through the dataset
        num_epoch times, printing the duration per epoch
        """
        lr = 0.001
        batch_size = 32
        # Labels for real data: 
        # - for discriminator, this is real images
        # - for generator this is what we wanted the discriminator output to be
        real_samples_labels = torch.ones(
            (batch_size, 1), device=GPU_DEVICE
        )
        # Init loss functions
        loss_function = nn.BCELoss()
        if model == "discriminator":
            self.generator.model.eval()
            self.discriminator.model.train()
            # total data is dataset * num_epochs
            num_epochs = 10
            # Load train data
            train_set = self.data()
            train_loader = torch.utils.data.DataLoader(
                train_set, batch_size=batch_size, shuffle=True
            )
            # Labels for generated data, all 0
            generated_samples_labels = torch.zeros(
                (batch_size, 1), device=GPU_DEVICE
            )
            # Load optimizer
            optimizer_discriminator = torch.optim.Adam(
                self.discriminator.parameters(), lr=lr,
            )
        else: # generator
            self.generator.model.train()
            self.discriminator.model.eval()
            # total data is batch_size * num_epochs
            num_epochs = 50
            # Load optimizer
            optimizer_generator = torch.optim.Adam(
                self.generator.parameters(), lr=lr,
            )
        loop_start = timeit.default_timer()
        # Repeat num_epoch times
        for epoch in range(num_epochs):
            # Iterate through dataset
            if model == "discriminator":
                for n, (images, labels) in enumerate(train_loader):
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
            else: # generator
                # Data for training the generator
                latent_space_samples = self.latent_input(batch_size)
                # Training the generator
                self.generator.zero_grad()
                generated_samples = self.generator(latent_space_samples)
                output_discriminator_generated = self.discriminator(
                    generated_samples
                )
                loss_generator = loss_function(
                    output_discriminator_generated, real_samples_labels
                )
                loss_generator.backward()
                optimizer_generator.step()
        if model == "discriminator":
            # Show loss
            print(f"Epoch: {epoch} Loss D.: {loss_discriminator}")
            print(timeit.default_timer() - loop_start)
        else: # generator
            print(f"Epoch: {epoch} Loss G.: {loss_generator}")
            print(timeit.default_timer() - loop_start)

    @staticmethod
    def latent_input(batch_size=1, generated=True):
        return torch.randn(
            batch_size, 1, 280, 280, device=GPU_DEVICE
        )


def main():
    # to train saved or load new model
    if False:
        gan = GAN.load("GAN_4198.6636476", mode="train")
    else:
        gan = GAN(Discriminator(), Generator())
    
    # Train the models
    start = timeit.default_timer()
    for i in range(11):
        gan.train(model="discriminator")
        gan.train(model="generator")
        if i % (x := 1) == 0:
            print("Train Time:")
            print(timeit.default_timer() - start)
            gan.save(f"models/GAN_280l_{timeit.default_timer()}")
            start = timeit.default_timer()

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
