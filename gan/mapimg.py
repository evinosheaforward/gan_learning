import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from skimage.transform import resize
import torch

# Use GPU switch (TODO: make this an arg ofc)
GPU_DEVICE = torch.device("cuda")  # Default CUDA device
# GPU_DEVICE = None

def download_image(url=''):
    url = "http://archive.wizards.com/default.asp?x=dnd/mwa/archiveall
  

if __name__ == "__main__":

    img=mpimg.imread('..\img2.jpg')

    print(img.shape)
    print(type(img))
    print(img.shape)
    img2 = torch.from_numpy(resize(img, (300, 400, 3)))
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