import matplotlib.pyplot as plt

from gan.mapmaker import models


def test_generator():
    """validate input / output sizes"""
    print(models.__file__)
    gan = models.GAN()
    output = gan.generator(gan.latent_input()).cpu()
    print(output)
    print(output.size())
    # plt.imshow(output[0, :, :, :])
    assert output.size() == (1, 3, 800, 800)


def test_discriminator():
    """validate input / output sizes"""
    gan = models.GAN()
    output = gan.discriminator(gan.discriminator_latent_input()).cpu()
    print(output)
    print(output.size())
    assert output.size() == (1, 1)
