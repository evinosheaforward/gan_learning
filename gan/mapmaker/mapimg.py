import os
import shutil

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from lxml import html
import requests

from skimage.transform import resize
import torch

# Use GPU switch (TODO: make this an arg ofc)
GPU_DEVICE = torch.device("cuda")  # Default CUDA device
# GPU_DEVICE = None


def download_image(url):
    basename = os.path.basename(url)
    path = os.path.join("data", "dnd_maps", basename)
    if os.path.exists(path):
        return
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        with open(path, 'wb') as f:
            r.raw.decode_content = True
            shutil.copyfileobj(r.raw, f)   
    return path


def download_dataset_mapoftheweek():
    url = "http://archive.wizards.com/default.asp?x=dnd/mwa/archiveall"
    lines = [
        line
        for line in requests.get(url).text.split("\n")
        if "Download 72dpi JPG" in line
    ]
    page = requests.get(url)
    webpage = html.fromstring(page.content)
    urls = [
        "http://archive.wizards.com" + imgpath
        for imgpath in webpage.xpath('//a/@href')
        if "mapofweek" in imgpath
        and ".jpg" in imgpath
    ]
    for url in urls:
        print(url)
        try:
            rpath = download_image(url)
        except:
            pass
    return rpath


def plot_img(fpath):
    img = mpimg.imread(fpath)
    print(img.shape)
    print(type(img))
    print(img.shape)
    img2 = torch.from_numpy(resize(img, (800, 800, 3)))
    print(type(img2))
    print(img2.shape)
    plt.imshow(img)
    plt.figure()
    plt.imshow(img2)
    plt.show()

    # from PIL import Image
    # image.show()


if __name__ == "__main__":
    fpath = download_dataset_mapoftheweek()
    print(len(os.listdir("data/dnd_maps/")))
    plot_img(fpath)
