import torch
import torch.nn.functional as F
from datetime import datetime
import logging
import argparse
import os
import numpy as np

from model.RISNet import RISNet
from utils.dataloader import get_loader, test_dataset
from torch.utils.tensorboard import SummaryWriter
from utils.utils import adjust_lr, clip_gradient

best_mae = 1
best_epoch = 1


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()


def train(train_loader, model, optim, epoch, opt, total_step, writer):
    total_loss = 0
    model.train()
    for step, data in enumerate(train_loader):
        imgs, gts, depths = data
        imgs = imgs.cuda()
        gts = gts.cuda()
        depths = depths.cuda()
        input_fea = torch.cat((imgs, depths), dim=0)

        optim.zero_grad()
        stage_pre, pre = model(input_fea)
        stage_loss_list = [structure_loss(out, gts) for out in stage_pre]
        stage_loss = 0
        gamma = 0.2
        for iteration in range(len(stage_pre)):
            stage_loss += (gamma * iteration) * stage_loss_list[iteration]

        map_loss = structure_loss(pre, gts)
        loss = stage_loss + map_loss
        loss.backward()
        clip_gradient(optimizer, opt.clip)
        optim.step()

        total_loss += loss

        if step % 20 == 0 or step == total_step:
            print(
                '[{}] => [Epoch Num: {:03d}/{:03d}] => [Global Step: {:04d}/{:04d}] => [Loss: {:0.4f}]'.
                format(datetime.now(), epoch, opt.epoch, step, total_step, loss.item()))
            logging.info(
                '#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:0.4f}'.
                format(epoch, opt.epoch, step, total_step, loss.item()))

    writer.add_scalar("Train_Loss", total_loss, global_step=epoch)

    save_path = opt.save_path
    if epoch % opt.epoch_save == 0:
        torch.save(model.state_dict(), save_path + str(epoch) + '_RISNet.pth')



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=100, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='choosing optimizer Adam')
    parser.add_argument('--augmentation', default=False, help='choose to do random flip rotation')
    parser.add_argument('--batchsize', type=int, default=4, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=704, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
    parser.add_argument('--train_path', type=str, default='your dataset path', help='path to train dataset')
    parser.add_argument('--save_path', type=str, default='your save path', help='path to save your model')
    parser.add_argument('--epoch_save', type=int, default=5, help='every n epochs to save model')
    opt = parser.parse_args()

    os.makedirs(opt.save_path, exist_ok=True)

    logging.basicConfig(filename=opt.save_path+'log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level = logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info("CCD-Train")

    model = RISNet().cuda()
    if opt.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), opt.lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.Adam(model.parameters(), opt.lr)

    train_image_root = '{}/Imgs/'.format(opt.train_path)
    train_gt_root = '{}/GT/'.format(opt.train_path)
    train_depth_root = '{}/Depth/'.format(opt.train_path)
    train_loader = get_loader(train_image_root, train_gt_root, train_depth_root, batch_size=opt.batchsize, image_size=opt.trainsize, num_workers=20)

    total_step = len(train_loader)

    writer = SummaryWriter(opt.save_path + "SummaryWriter")

    print('--------------------training----------------------')

    for epoch in range(1, opt.epoch+1):

        adjust_lr(optimizer, epoch, opt.decay_rate, opt.decay_epoch)

        train(train_loader, model, optimizer, epoch, opt, total_step, writer)

    writer.close()