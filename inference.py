import torch
import torch.optim
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def dehaze_image(image_name):
    data_hazy = Image.open(image_name)
    data_hazy = np.array(data_hazy) / 255.0
    original_img = data_hazy.copy()

    data_hazy = torch.from_numpy(data_hazy).float()
    data_hazy = data_hazy.permute(2, 0, 1)
    data_hazy = data_hazy.unsqueeze(0)
    bolb = data_hazy

    dehaze_net = torch.load('saved_models/dehaze_net_epoch_17.pth', map_location=torch.device('cpu'))


    a = dehaze_net(data_hazy)
    b = a.detach()
    c = b.numpy()
    d =c.squeeze()
    clean_image = dehaze_net(data_hazy).detach().numpy().squeeze()
    clean_image = np.swapaxes(clean_image, 0, 1)
    clean_image = np.swapaxes(clean_image, 1, 2)

    plt.subplot(1, 2, 1)
    plt.imshow(original_img)
    plt.axis('off')
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(clean_image)
    plt.axis('off')
    plt.title('Dehaze Image')
    plt.show()


if __name__ == '__main__':
    img_name = './test_images/test0.png'
    dehaze_image(img_name)
