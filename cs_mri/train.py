import torch
import random
from math import log10
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from torch.optim.lr_scheduler import StepLR

from model import CS_MRI_Network
from dataset import TrainingDataset

pretrainedModel = ''
batchSize = 4
epoch_num = 1500

torch.manual_seed(1000)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = CS_MRI_Network().to(device)

if pretrainedModel != '':
    net.load_state_dict(torch.load(pretrainedModel))

criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(net.parameters(), lr=0.005, betas=(0.9, 0.999), weight_decay=0.0001)
scheduler = StepLR(optimizer, step_size=600, gamma=0.5)

train_path = "./data/train_ori"
val_path = "./data/val_ori"
mask_path = './data/mask'
train_dataset = TrainingDataset(train_path, mask_path)
val_dataset = TrainingDataset(val_path, mask_path)
train_dataloader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True, collate_fn=train_dataset.collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=batchSize, shuffle=False, collate_fn=val_dataset.collate_fn)

loss_track = []
for epoch in range(epoch_num):
    train_loss, val_loss, avg_psnr = 0.0, 0.0, 0.0
    
    net.train()
    for i, batch in enumerate(train_dataloader):
        input_data, label = batch[0].to(device), batch[1].to(device)
        
        optimizer.zero_grad()
        output = net(input_data)
        loss = criterion(output[:,:,:,:,0], label)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    
    net.eval()
    with torch.no_grad():
        for batch in val_dataloader:
            input_data_v, label_v = batch[0].to(device), batch[1].to(device)
            
            output_v = net(input_data_v)
            loss = criterion(output_v[:,:,:,:,0], label_v)
            
            val_loss += loss.item()
            avg_psnr += 10 * log10(1 / loss.item())
        avg_psnr /= len(val_dataloader)
    
    scheduler.step()
    loss_track.append((train_loss, val_loss, avg_psnr))
    torch.save(loss_track, 'loss.pth')
    
    print('[{:4d}/{}] train_loss: {:.3f}, eval_loss: {:.3f}, Avg. PSNR: {:.4f} dB'.format(epoch+1, epoch_num, train_loss, val_loss, avg_psnr))
    
    if epoch % 10 == 0:
        torch.save(net.state_dict(), 'checkpoint/epoch_{}.pth'.format(epoch+1))
        
    if epoch % 5 == 0:
        vutils.save_image(torch.cat((torch.ifft(input_data, 2)[:,:,:,:,0].detach(), torch.ifft(input_data_v, 2)[:,:,:,:,0].detach(), \
        output[:,:,:,:,0].detach(), output_v[:,:,:,:,0].detach(), label, label_v), 0), 'CS_MRI_results/epoch_{}.png'.format(epoch+1), nrow=batchSize*2)