import torch

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from gan.mapmaker import mapimg
from gan.mapmaker import models


def test_load_data():
    """Load whole dataset, long test"""
    data = mapimg.load_images("data/dnd_maps", one=True)
    maximum = torch.max(data)
    minimum = torch.min(data)
    print(data)
    print(minimum, maximum)
    assert maximum < 1
    assert maximum > 0
    assert minimum > -1
    assert minimum < 0
    # plt.imshow((data[0, :, :, :].permute(1, 2, 0) + 1.0) / 2.0)
    # plt.show()
    print(data.size())
    assert data.size() == (1, 3, 128, 128)


def test_add_noise():
    data = mapimg.load_images("data/dnd_maps", one=True)
    data = models.add_noise(data.cuda()).cpu()
    plt.imshow((data[0, :, :, :].permute(1, 2, 0) + 1.0) / 2.0)
    plt.show()
    maximum = torch.max(data)
    minimum = torch.min(data)
    print(data)
    print(minimum, maximum)
    assert maximum < 1
    assert maximum > 0
    assert minimum > -1
    assert minimum < 0
    print(data.size())
    assert data.size() == (1, 3, 128, 128)
