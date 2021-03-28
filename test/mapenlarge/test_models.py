import torch

from gan.mapenlarge import models
from gan.mapenlarge import mapimg


def test_generator():
    """validate input / output sizes"""
    print(models.__file__)
    gan = models.MapEnlarge()
    input = gan.latent_input()
    maximum = torch.max(input)
    minimum = torch.min(input)
    print(minimum, maximum)
    assert maximum <= 2
    assert maximum > 0
    assert minimum >= -2
    assert minimum < 0
    output = gan.generator(input).cpu()
    print(output)
    print(output.size())
    import matplotlib.pyplot as plt
    plt.imshow((output[0, :, :, :].detach().permute(1, 2, 0) +1) / 2)
    plt.show()
    maximum = torch.max(output)
    minimum = torch.min(output)
    print(minimum, maximum)
    assert maximum <= 1
    assert maximum > 0
    assert minimum >= -1
    assert minimum < 0
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
    maximum = torch.max(output)
    minimum = torch.min(output)
    print(minimum, maximum)
    assert maximum <= 1
    assert minimum >= 0
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
    maximum = torch.max(output)
    minimum = torch.min(output)
    print(minimum, maximum)
    assert maximum <= 1
    assert minimum >= 0
    assert output.size() == (1, 1)
