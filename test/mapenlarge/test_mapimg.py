import torch

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from gan.mapenlarge import mapimg
from gan.mapenlarge import models


# def test_load_images_torchvision():
#     """Load whole dataset, long test"""
#     data = mapimg.load_images_torchvision("data/dnd_maps")
#     maximum = torch.max(data)
#     minimum = torch.min(data)
#     print(data)
#     print(minimum, maximum)
#     assert maximum <= 1
#     # assert maximum > 0
#     assert minimum >= 0
#     # assert minimum < 0
#     plt.imshow(data[0, :, :, :].permute(1, 2, 0) )
#     plt.show()
#     print(data.size())
#     assert data.size() == (250, 3, 1024, 1024)

def test_load_data():
    """Load whole dataset, long test"""
    data = mapimg.load_images("data/dnd_maps", num=5)
    maximum = torch.max(data)
    minimum = torch.min(data)
    print(data)
    print(minimum, maximum)
    assert maximum <= 1
    assert maximum > 0
    assert minimum >= -1
    assert minimum < 0
    # plt.imshow((data[0, :, :, :].permute(1, 2, 0) )
    # plt.show()
    print(data.size())
    assert data.size() == (5, 3, 512, 512)


def test_add_noise():
    data = mapimg.load_images("data/dnd_maps", num=1)
    data = models.add_noise(data.cuda()).cpu()
    # plt.imshow((data[0, :, :, :].permute(1, 2, 0) )
    # plt.show()
    maximum = torch.max(data)
    minimum = torch.min(data)
    print(data)
    print(minimum, maximum)
    assert maximum <= 1
    assert maximum > 0
    assert minimum >= -1
    assert minimum < 0
    print(data.size())
    assert data.size() == (1, 3, 512, 512)
    print(data.element_size(), data.nelement())
