import numpy as np
from PIL import Image
import onnxruntime as ort
import matplotlib.pyplot as plt

def dehaze_image(image_name):
    data_hazy = Image.open(image_name)
    data_hazy = np.array(data_hazy) / 255.0
    original_img = data_hazy.copy()

    data_hazy = np.array(data_hazy, dtype=np.float32)
    data_hazy = np.transpose(data_hazy, (2, 0, 1))
    data_hazy = np.expand_dims(data_hazy, 0)
    bolb = data_hazy

    input_name = 'input'
    output_name = 'output'
    dehaze_net = ort.InferenceSession("./dehaze_net.onnx", providers=ort.get_available_providers())
    netOutputImg = dehaze_net.run([output_name], {input_name: bolb})
    pdata = netOutputImg[0]
    # clean_image = netOutputImg.squeeze()
    clean_image = pdata.squeeze()
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
