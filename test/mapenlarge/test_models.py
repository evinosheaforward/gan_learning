from gan.mapenlarge import models
from gan.mapenlarge import mapimg


def test_generator():
    """validate input / output sizes"""
    print(models.__file__)
    gan = models.MapEnlarge()
    output = gan.generator(gan.latent_input()).cpu()
    print(output)
    print(output.size())
    # import matplotlib.pyplot as plt
    # plt.imshow(output[0, :, :, :].detach().permute(1, 2, 0))
    # plt.show()
    assert output.size() == (1, 3, 512, 512)


def test_discriminator_generator():
    """validate input / output sizes"""
    gan = models.MapEnlarge()
    input_data = gan.discriminator_latent_input()
    print(input_data.size())
    assert input_data.cpu().size() == (1, 3, 512, 512)
    print(input_data.type())
    output = gan.discriminator(input_data).cpu()
    print(output)
    print(output.size())
    assert output.size() == (1, 1)


def test_discriminator_corpus():
    """validate input / output sizes"""
    gan = models.MapEnlarge()
    input_data = mapimg.load_images("data/dnd_maps", num=1).cuda()
    print(input_data.size())
    print(input_data.type())
    assert input_data.cpu().size() == (1, 3, 512, 512)
    output = gan.discriminator(input_data).cpu()
    print(output)
    print(output.size())
    assert output.size() == (1, 1)


