import torch
import random
from math import log10
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import torchvision.transforms.functional as TF
from torch.optim.lr_scheduler import StepLR
import argparse

from model import PGDeblurringNetwork
from dataset import TrainingDataset

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='')
parser.add_argument('--lr', type=float, default=0.001)
args = parser.parse_args()

pretrainedModel = args.model
lr = args.lr
batchSize = 8
epoch_num = 1500

torch.manual_seed(1000)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = PGDeblurringNetwork().to(device)

criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.0001)
scheduler = StepLR(optimizer, step_size=300, gamma=0.5)
loss_track = []

if pretrainedModel != '':
    model = torch.load(pretrainedModel)
    net.load_state_dict(model['state_dict'])
    if lr == 0.001 :
        optimizer.load_state_dict(model['optimizer'])
        scheduler.load_state_dict(model['scheduler'])
        loss_track = model['loss']
    epoch0 = model['epoch']
else: 
    epoch0 = 0

train_path = "../../dataset/BSDS500/train"
val_path = "../../dataset/BSDS500/val"
train_dataset = TrainingDataset(train_path)
val_dataset = TrainingDataset(val_path)
train_dataloader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True, collate_fn=train_dataset.collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=batchSize, shuffle=False, collate_fn=val_dataset.collate_fn)

loss_track = []
for epoch in range(epoch0, epoch_num):
    train_loss, val_loss, avg_psnr = 0.0, 0.0, 0.0
    
    net.train()
    for i, batch in enumerate(train_dataloader):
        img, noisyImg = batch[0].to(device), batch[1].to(device)

        optimizer.zero_grad()
        output = net(noisyImg)
        loss = criterion(output, img)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    
    net.eval()
    with torch.no_grad():
        for batch in val_dataloader:
            img_v, noisyImg_v = batch[0].to(device), batch[1].to(device)
            
            output_v = net(noisyImg_v)
            loss = criterion(output_v, img_v)
            
            val_loss += loss.item()
            avg_psnr += 10 * log10(1 / loss.item())
        avg_psnr /= len(val_dataloader)
    
    scheduler.step()
    loss_track.append((train_loss, val_loss, avg_psnr))
    torch.save(loss_track, 'loss.pth')
    
    print('[{:4d}/{}] lr: {:.5f}, train_loss: {:.3f}, eval_loss: {:.3f}, Avg. PSNR: {:.4f} dB'.format(epoch+1, epoch_num, optimizer.param_groups[0]['lr'], train_loss, val_loss, avg_psnr))
    
    if epoch % 10 == 0:
        torch.save({
            'epoch': epoch+1,
            'state_dict':net.state_dict(),
            'optimizer':optimizer.state_dict(),
            'scheduler':scheduler.state_dict(),
            'loss':loss_track
        }, 'checkpoint/epoch_{}.pth'.format(epoch+1))
        # print(net.C0.data)
        # print(net.Ck.data)
        
    if epoch % 5 == 0:
        vutils.save_image(torch.cat((noisyImg.detach(), noisyImg_v.detach(), output.detach(), output_v.detach(), img, img_v), 0), 'deblurringImg/epoch_{}.png'.format(epoch+1), nrow=12)
