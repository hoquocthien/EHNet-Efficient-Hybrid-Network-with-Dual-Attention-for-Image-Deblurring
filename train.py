import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
torch.backends.cudnn.benchmark = True

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import random
import time
import numpy as np

from ptflops import get_model_complexity_info
import utils
from data_RGB import get_training_data, get_validation_data
from EHNet import EHNet as myNet
import losses
from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm
import argparse

######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

start_epoch = 1

parser = argparse.ArgumentParser(description='Image Deblurring')

parser.add_argument('--train_dir', default='./Dataset/GoPro/train/', type=str, help='Directory of train images')
parser.add_argument('--val_dir', default='./Dataset/GoPro/test/', type=str, help='Directory of validation images')
parser.add_argument('--model_save_dir', default='./checkpoints/', type=str, help='Path to save weights')
parser.add_argument('--pretrain_weights', default='./checkpoints/models/GoPro/model_latest.pth', type=str, help='Path to pretrain-weights')

parser.add_argument('--session', default='GoPro', type=str, help='session')
parser.add_argument('--patch_size', default=256, type=int, help='patch size')
parser.add_argument('--num_epochs', default=300, type=int, help='num_epochs')
parser.add_argument('--batch_size', default=2, type=int, help='batch_size')
parser.add_argument('--val_epochs', default=10, type=int, help='val_epochs')
args = parser.parse_args()

session = args.session
patch_size = args.patch_size

model_dir = os.path.join(args.model_save_dir, 'models',  session)
utils.mkdir(model_dir)

train_dir = args.train_dir
val_dir = args.val_dir

num_epochs = args.num_epochs
batch_size = args.batch_size
val_epochs = args.val_epochs

start_lr = 2e-4
end_lr = 1e-7

######### Model ###########
model_restoration = myNet()
model_restoration.cuda()

device_ids = [i for i in range(torch.cuda.device_count())]
if torch.cuda.device_count() > 1:
  print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")

macs, params = get_model_complexity_info(model_restoration, (3, 256, 256), as_strings=False,
                                        print_per_layer_stat=False, verbose=True)
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params)) 

optimizer = optim.Adam(model_restoration.parameters(), lr=start_lr, betas=(0.9, 0.999), eps=1e-8)

######### Scheduler ###########
warmup_epochs = 3
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs-warmup_epochs, eta_min=end_lr)
scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)

RESUME = False
Pretrain = False
model_pre_dir = args.pretrain_weights
######### Pretrain ###########
if Pretrain:
    utils.load_checkpoint(model_restoration, model_pre_dir)

    print('------------------------------------------------------------------------------')
    print("==> Retrain Training with: " + model_pre_dir)
    print('------------------------------------------------------------------------------')

######### Resume ###########
if RESUME:
    path_chk_rest = utils.get_last_path(model_dir, '_latest.pth')
    utils.load_checkpoint(model_restoration,path_chk_rest)
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1
    utils.load_optim(optimizer, path_chk_rest)

    for i in range(1, start_epoch):
        scheduler.step()
    new_lr = scheduler.get_lr()[0]
    print('------------------------------------------------------------------------------')
    print("==> Resuming Training with learning rate:", new_lr)
    print('------------------------------------------------------------------------------')

if len(device_ids)>1:
    model_restoration = nn.DataParallel(model_restoration, device_ids=device_ids)

######### Loss ###########
criterion_fft = losses.fftLoss()
criterion_l1 = torch.nn.L1Loss()
######### DataLoaders ###########
train_dataset = get_training_data(train_dir, {'patch_size':patch_size})
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=False, pin_memory=True)

val_dataset = get_validation_data(val_dir, {'patch_size':patch_size})
val_loader = DataLoader(dataset=val_dataset, batch_size=2, shuffle=False, num_workers=8, drop_last=False, pin_memory=False)

print('===> Start Epoch {} End Epoch {}'.format(start_epoch, num_epochs + 1))
print('===> Loading datasets')

best_psnr = 0
best_epoch = 0
iter = 0

for epoch in range(start_epoch, num_epochs + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    train_id = 1

    model_restoration.train()
    for i, data in enumerate(tqdm(train_loader), 0):

        
        for param in model_restoration.parameters():
            param.grad = None

        target_ = data[0].cuda()
        input_  = data[1].cuda()
        
        target = target_
        restored = model_restoration(input_)
        loss_l1 = criterion_l1(restored, target) 
        loss_fft = criterion_fft(restored, target) 
        
        loss = loss_l1 + 0.1 * loss_fft
        loss.backward()
        optimizer.step()
        epoch_loss +=loss.item()
        iter += 1
        
    #### Evaluation ####
    if epoch % val_epochs == 0:
        model_restoration.eval()
        psnr_val_rgb = []
        for ii, data_val in enumerate(tqdm(val_loader), 0):
            target = data_val[0].cuda()
            input_ = data_val[1].cuda()

            with torch.no_grad():
                restored = model_restoration(input_)

            for res,tar in zip(restored, target):
                psnr_val_rgb.append(utils.torchPSNR(res, tar))

        psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()
        
        if psnr_val_rgb > best_psnr:
            best_psnr = psnr_val_rgb
            best_epoch = epoch
            torch.save({'epoch': epoch, 
                        'state_dict': model_restoration.state_dict(),
                        'optimizer' : optimizer.state_dict()
                        }, os.path.join(model_dir,"model_best.pth"))

        print("[epoch %d PSNR: %.4f --- best_epoch %d Best_PSNR %.4f]" % (epoch, psnr_val_rgb, best_epoch, best_psnr))


        with open('results.txt', 'a') as f:
            f.write('epoch: {} PSNR:{:.4f}'.format(epoch, psnr_val_rgb))

        torch.save({'epoch': epoch, 
                    'state_dict': model_restoration.state_dict(),
                    'optimizer' : optimizer.state_dict()
                    }, os.path.join(model_dir,f"model_epoch_{epoch}.pth")) 

    scheduler.step()
    
    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time, epoch_loss, scheduler.get_lr()[0]))
    print("------------------------------------------------------------------")

    torch.save({'epoch': epoch, 
                'state_dict': model_restoration.state_dict(),
                'optimizer' : optimizer.state_dict()
                }, os.path.join(model_dir,"model_latest.pth")) 


