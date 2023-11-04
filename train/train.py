
import torch
import os 
import sys
# print(sys.path)
sys.path.append("..")
from model.PLS3D import *
from torch.utils.data import DataLoader, Dataset
from tqdm.notebook import trange, tqdm 
from optparse import OptionParser
import numpy as np
import apex.amp as amp
import glob
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torchvision import models
from torch import optim
# from utils.loss import *
from trannier import *
dataset_type=''
batch_size=1 #fixed to 1

gpu = 1
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
cuda = torch.cuda.is_available()
print('Cuda: {}'.format(cuda))
configurations = dict(
    epochs=100,
    lr=0.001,
    gpu=True,
    growth_rate=12,
    dataset_type='',
    a=0.2,
    b=0.8,
    load_dir = None,
    save_path = './saves',
    dataroot='',# path you put the data
    lock_param=True,
             
)

resume = ''
print(configurations)

net = NMODEPls_Net_3D_cp_multitask(n_channels=1, n_classes=6,g = 12)
net.eval()
net.to('cuda')

# lossa = Multi_label_DiceLoss()
lossa = nn.CrossEntropyLoss()
lossb = MultiLabel_Focal_loss(2,1)
criterion = Multi_loss(a=0.6,b=0.4,lossa=lossa,lossb=lossb)
criterion_aux = nn.CrossEntropyLoss()
t = time.gmtime()
t = time.strftime("%Y%m%d%H%M%S",t)
store_1 = 'log'+t
store_path = os.path.join(configurations['save_path'],store_1)
cpt_store_path = os.path.join(store_path,'checkpoints')
if os.path.exists(cpt_store_path) is False:
    os.makedirs(cpt_store_path)
log_store_path = os.path.join(store_path,'logs')
if os.path.exists(log_store_path) is False:
    os.makedirs(log_store_path)


try:
    train_plsnet(
        net=net,
        epochs=configurations['epochs'],
        batch_size=1,
        lr=configurations['lr'],
        gpu=configurations['gpu'],
        multi_gpu=False,
        load_dir = configurations['load_dir'],
        growth_rate=configurations['growth_rate'],
        dataset_type = configurations['dataset_type'],
        save_path = store_path                                                     ,
        a = float(configurations['a']),
        b = float(configurations['b']),
        dataroot=configurations['dataroot'],
        store_path = store_path,
        cpt_store_path=cpt_store_path,
        log_store_path=log_store_path,
        t = t,
        criterion=criterion,
        )
except KeyboardInterrupt:
    torch.save(net.state_dict(), os.path.join(store_path,'INTERRUPTED.pth'))
    print('Saved interrupt')
    try:
        sys.exit(0)
    except SystemExit:
        os._exit(0)