from gan.mapmaker.models import GAN


def test_generator():
    gan = GAN()
    assert gan.generator(gan.latent_input()) == (3, 800, 800)


def test_discriminator():
    gan = GAN()
    assert gan.discriminator(gan.latent_input()).size() == (1,)
