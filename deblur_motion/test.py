import torch
import random
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import argparse
from math import log10
from model import PGDeblurringNetwork
from dataset import TestDataset

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='')
parser.add_argument('--input', type=str, default='')
args = parser.parse_args()

batchSize = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = PGDeblurringNetwork().to(device)
criterion = nn.MSELoss().to(device)

model = torch.load(args.model)
net.load_state_dict(model['state_dict'])

test_dataset = TestDataset(args.input)
test_dataloader = DataLoader(test_dataset, batch_size=batchSize, shuffle=True, collate_fn=test_dataset.collate_fn)

net.eval()
with torch.no_grad():
    batch = next(iter(test_dataloader))
    img_v, noisyImg_v = batch[0].to(device), batch[1].to(device)
    output_v = net(noisyImg_v)
    loss = criterion(output_v, img_v)
    avg_psnr = 10 * log10(1 / loss.item())
    print('psnr: {:.4f} dB'.format(avg_psnr))
    vutils.save_image(torch.cat((noisyImg_v.detach(), output_v.detach(), img_v), 0), 'test.png', nrow=batchSize)
