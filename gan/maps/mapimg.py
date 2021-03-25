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


def download_image(url=""):
    pass


def download_dataset_mapoftheweek():
    url = "http://archive.wizards.com/default.asp?x=dnd/mwa/archiveall"
    lines = [
        # line.split('href="')[0].split('"')[0]
        line
        for line in requests.get(url).text.split("\n")
        if "Download 72dpi JPG" in line
    ]
    page = requests.get(url)
    # print(page.content)
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
            basename = os.path.basename(url)
            path = os.path.join("data", "dnd_maps", basename)
            if os.path.exists(path):
                continue
            r = requests.get(url)
            if r.status_code == 200:
                with open(path, 'wb') as f:
                    r.raw.decode_content = True
                    shutil.copyfileobj(r.raw, f)   
        except:
            pass

    # print(lines)
    # print(requests.get(url, headers={"referer": lines[0]}).text)


def plot_img(fpath):
    img = mpimg.imread(fpath)
    print(img.shape)
    print(type(img))
    print(img.shape)
    img2 = torch.from_numpy(resize(img, (800, 800, 3)))
    print(type(img2))
    print(img2.shape)
    if True:
        plt.imshow(img)
        plt.figure()
        plt.imshow(img2)
        plt.show()

    # from PIL import Image
    # image = Image.open("..\img1.jpg")
    # image.show()


if __name__ == "__main__":
    # download_dataset_mapoftheweek()
    fpath = os.path.join(
        os.path.abspath("."),
        'data',
        'dnd_maps',
        '05_May2007_300_48fjf2_ppi.jpg',
    )
        #castle.png')
    print(len(os.listdir("data/dnd_maps/")))
    plot_img(fpath)
