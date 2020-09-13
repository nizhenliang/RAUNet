import torch
import argparse
from torch.utils.data import DataLoader
from torch import autograd, optim
import os
import numpy as np
import tqdm
import datetime
import math
from validation import val_multi
import glob
from load_dataset import Load_Dataset
from tensorboardX import SummaryWriter
import torch.nn as nn
from loss import CELDice
from RAUNet import RAUNet

device_ids = [0]

parse=argparse.ArgumentParser()
num_classes=11
lra=0.00004

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lra* (0.8 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def load_filename():
    train_file_names = glob.glob('dataset/train/images/*.png')
    val_file_names = glob.glob('dataset/test/images/*.png')
    return train_file_names,val_file_names

def train():
    mod = RAUNet(num_classes=num_classes,pretrained=True)

    model = mod.cuda(device_ids[0])
    model = nn.DataParallel(model, device_ids=device_ids)

    batch_size = args.batch_size
    criterion = CELDice(0.2, num_classes=num_classes)
    optimizer = optim.Adam(model.parameters(), lr=lra)

    train_file, val_file = load_filename()
    liver_dataset = Load_Dataset(train_file)
    val_dataset= Load_Dataset(val_file)

    dataloaders = DataLoader(liver_dataset, batch_size=batch_size, shuffle=True, num_workers=12) # drop_last=True
    val_load=DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=12)
    train_model(model, criterion, optimizer, dataloaders, val_load, num_classes)

def train_model(model, criterion, optimizer, dataload,val_load,num_classes,num_epochs=200):
    loss_list=[]
    dice_list=[]
    logs_dir = 'Logs/T{}/'.format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.mkdir(logs_dir)
    writer = SummaryWriter(logs_dir)
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        # print('-' * 10)
        dt_size = len(dataload.dataset)
        tq = tqdm.tqdm(total=math.ceil(dt_size/args.batch_size))
        tq.set_description('Epoch {}'.format(epoch))
        epoch_loss =[]
        step = 0
        for x, y in dataload:
            step += 1
            inputs = x.cuda(device_ids[0])
            y=y.long()
            labels = y.cuda(device_ids[0])
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            tq.update(1)
            epoch_loss.append(loss.item())
            epoch_loss_mean = np.mean(epoch_loss).astype(np.float64)
            tq.set_postfix(loss='{0:.3f}'.format(epoch_loss_mean))
        loss_list.append(epoch_loss_mean)
        tq.close()
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss_mean))
        dice, iou =val_multi(model, criterion, val_load, num_classes,args.batch_size,device_ids)
        writer.add_scalar('Loss', epoch_loss_mean, epoch)
        writer.add_scalar('Dice', dice, epoch)
        writer.add_scalar('IoU', iou, epoch)
        dice_list.append([dice,iou])
        adjust_learning_rate(optimizer, epoch)
        torch.save(model.module.state_dict(), logs_dir + 'weights_{}.pth'.format(epoch))
        fileObject = open(logs_dir+'LossList.txt', 'w')
        for ip in loss_list:
            fileObject.write(str(ip))
            fileObject.write('\n')
        fileObject.close()
        fileObject = open(logs_dir + 'dice_list.txt', 'w')
        for ip in dice_list:
            fileObject.write(str(ip))
            fileObject.write('\n')
        fileObject.close()

    writer.close()
    return model


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("--batch_size", type=int, default=8)
    parse.add_argument("--ckp", type=str, help="the path of model weight file")
    args = parse.parse_args()
    train()
