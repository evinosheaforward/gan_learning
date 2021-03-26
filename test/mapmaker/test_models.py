from gan.mapmaker import models


def test_generator():
    """validate input / output sizes"""
    print(models.__file__)
    gan = models.GAN()
    output = gan.generator(gan.latent_input()).cpu()
    print(output)
    print(output.size())
    # import matplotlib.pyplot as plt
    # plt.imshow(output[0, :, :, :].detach().permute(1, 2, 0))
    # plt.show()
    assert output.size() == (1, 3, 128, 128)


def test_discriminator():
    """validate input / output sizes"""
    gan = models.GAN()
    input_data = gan.discriminator_latent_input()
    print(input_data.cpu().size())
    output = gan.discriminator(input_data).cpu()
    print(output)
    print(output.size())
    assert output.size() == (1, 1)

def test_train_loop():
    pass