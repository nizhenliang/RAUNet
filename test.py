import torch
import argparse
from torch.utils.data import DataLoader
from load_dataset import Load_Dataset
from validation import val_multi
import glob
from loss import CELDice
from RAUNet import RAUNet


device_ids = [0]
parse=argparse.ArgumentParser()
weight_load = 'Logs/T20200913_164608/weights_0.pth'

num_classes=11

def test():
    device = torch.device("cpu")
    model = RAUNet(num_classes=num_classes, pretrained=True)
    model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(weight_load, map_location=device).items()})

    model=model.cuda(device_ids[0])
    criterion = CELDice(0.2, num_classes=num_classes)

    val_file_names = glob.glob('dataset/test/images/*.png')
    val_dataset = Load_Dataset(val_file_names)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,num_workers=16)

    val_multi(model, criterion, val_loader, num_classes,batch_size=args.batch_size,device_ids=device_ids)


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("--batch_size", type=int, default=8)
    args = parse.parse_args()
    test()