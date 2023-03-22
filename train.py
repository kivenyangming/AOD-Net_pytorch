import torch
import torch.optim
import torch.nn as nn
from model import AODNet
from data import MyDataset
from utils import weights_init
from torch.utils.data import DataLoader


def train(orig_images_path, hazy_images_path, batch_size, epochs):
    train_dataset = MyDataset(orig_images_path, hazy_images_path, mode='train')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # val_dataset = MyDataset(orig_images_path, hazy_images_path, mode='val')
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    dehaze_net = AODNet().cuda()
    dehaze_net.apply(weights_init)
    criterion = nn.MSELoss().cuda()
    optimizer = torch.optim.Adam(dehaze_net.parameters(), lr=1e-3, weight_decay=1e-4)

    dehaze_net.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for iteration, (img_orig, img_haze) in enumerate(train_loader):

            img_orig = img_orig.cuda()
            img_haze = img_haze.cuda()
            clean_image = dehaze_net(img_haze)
            loss = criterion(clean_image, img_orig)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(dehaze_net.parameters(), 0.1)
            optimizer.step()

            epoch_loss += loss.item()

        print('EPOCH : %04d  LOSS : %2.3f' % (epoch, epoch_loss / len(train_loader)))

        torch.save(dehaze_net(), './saved_models/dehaze_net_epoch_%d.pth' % epoch)


if __name__ == '__main__':
    orig_images_path = './data/gt/'
    hazy_images_path = './data/hazy/'
    batch_size = 12
    epochs = 30
    train(orig_images_path, hazy_images_path, batch_size, epochs)
