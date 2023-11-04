import torch
import os 
import sys
sys.path.append("..")
from dataset.dataLoader import Lobe_Dataset_multitask_20200318,Lobe_Dataset_Resampled
from torch.utils.data import DataLoader, Dataset
from tqdm.notebook import trange, tqdm 
from optparse import OptionParser
import numpy as np
import apex.amp as amp
import glob
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
# import wandb
from torchvision import models
from torch import optim
from utils.loss import *
import time
from utils.metric import calculate_dice
import torch.nn.functional as F
import logging
from torch.cuda import amp
slicenum = 140

def train_plsnet(net,
              epochs=60,
              lr=0.01,
              val_percent=0.05,
              batch_size=1,
              save_cp=True,
              gpu=True,
              multi_gpu=False,
             load_dir = None,
             growth_rate=12,
             dataset_type = 'f1',
             save_path = './',
             dataroot='',
             val_interval=1,
             save_interval=4,
             pretrain=None,
             pretrain_layer=None,
             lock_param=False,
             cpt_store_path='./',
             log_store_path='./',
             store_path = './',
             criterion = None ,
             maskout = False,
             t = '0',
             a=0.8,
             b=0.2,
             c=0.1):
    logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename=os.path.join(log_store_path,t+'.log'),
                    filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    train_logger = logging.getLogger('train_logger')
    val_logger = logging.getLogger('val_logger')    
    
    
    
    train_set=Lobe_Dataset_Resampled(data_set_type=dataset_type + '_train',dataroot=dataroot,maskout=maskout,logger=train_logger)
    validation_set=Lobe_Dataset_Resampled(data_set_type=dataset_type + '_test',dataroot = dataroot,maskout=maskout,logger=val_logger)
    
    train_loader = DataLoader(train_set, batch_size=batch_size,shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=batch_size)
    
    


    train_logger.info('''
    Starting training:
        Epochs: {}
        Learning rate: {}
        Training size: {}
        Validation size: {}
        Checkpoints: {}
        CUDA: {}
        save_path: {}
    '''.format(epochs, 
               lr, 
               len(train_set),
               len(validation_set), 
               str(save_cp), 
               str(gpu),
               str(store_path),
               ))
    train_losses=[]
    val_losses = []
    # criterion = nn.CrossEntropyLoss()
    if criterion==None:
        criterion = Multi_loss_ce_dice()
    
    if load_dir:
        checkpoint = torch.load(load_dir)
        try:
            net.load_state_dict(checkpoint)
        except:
            net.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        #optimizer.load_state_dict(checkpoint['optimizer'])
        # amp.load_state_dict(checkpoint['amp'])
        train_logger.info('Model loaded from {}'.format(load_dir))
    if pretrain!=None:
        checkpoint = torch.load(pretrain)
        if pretrain_layer==None:
            state_dict = {k:v for k,v in checkpoint['model'].items() if not k.startswith('conv')}
        else:
            state_dict = {k:v for k,v in checkpoint['model'].items() if k.split('.')[0] in pretrain_layer}
        train_logger.info("load pretrianed state_dict:"+pretrain)
        net.load_state_dict(state_dict,strict=False)
    if pretrain!=None and pretrain_layer!=None and lock_param:
        layer_counter = 0
        for (name, module) in net.named_children():
            if name in pretrain_layer:
                for layer in module.children():
                    for param in layer.parameters():
                        param.requires_grad = False
            
                    train_logger.info('Layer "{}" in module "{}" was frozen!'.format(layer_counter, name))
                    layer_counter+=1
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr,weight_decay=5e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor=0.6, patience=30)
    # opt_level = 'O2'
    # net, optimizer = amp.initialize(net, optimizer, opt_level=opt_level)
    scalar = amp.GradScaler()
    best_loss = 0
    best_val=0
    Save_flag=0
    total_validate_time = 0
    num=0
    for epoch in tqdm(range(epochs),total = epochs):
        train_logger.info('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        save_cp = False
        net.train()
        epoch_loss=0
        pbar = tqdm(train_loader, total=len(train_set))
        count = 0
        for data in pbar:
            optimizer.zero_grad()
            x_train,y_train=data
            num+=1
            count+=1
            x_train=x_train.type(torch.FloatTensor)
            y_train=y_train.type(torch.LongTensor)
            # aux_train = aux_train.type(torch.FloatTensor)
            if torch.cuda.is_available():
                inputs = x_train.to('cuda')
                target = y_train.to('cuda')
                # aux = aux_train.to('cuda')
            else:
                inputs = x_train
                target = y_train
            with amp.autocast():
                out1 = net(inputs)    
                loss = criterion(out1, target)
                epoch_loss += loss.item()
            
                del inputs,target,out1
        
                scalar.scale(loss).backward()
                scalar.step(optimizer)
                scalar.update()
                scheduler.step(loss)
            train_logger.info('Epoch : [{}/{}], loss: {:.4f}  '.format(
                epoch + 1,
                epochs,
                loss.item())+str(count)+'/'+str(int(len(train_set)/batch_size))+'  ')
            # break
        train_logger.info('Epoch finished ! Loss: {}'.format(batch_size*epoch_loss / len(train_set)))
        
        #validation
        if (epoch+1)%val_interval==0: #验证
            net.eval()
            dice=0
            val_loss = 0
            valtime = 0
            rm = []
            ru = []
            rl = []
            lu = []
            ll = []
            stat = {'1' : rm,
                    '2' : ru,
                    '3' : rl,
                    '4' : lu,
                    '5' : ll,}
            avg = []
            val_count = 0
            for data in tqdm(validation_loader):
                val_count+=1
                x_train,y_train=data
                x_train=x_train.type(torch.FloatTensor)
                y_train=y_train.type(torch.LongTensor)
                # aux_train = aux_train.type(torch.FloatTensor)
                
                if torch.cuda.is_available():
                    inputs = x_train.cuda()
                    target = y_train.cuda()
                    # aux = aux_train.cuda()
                else:
                    inputs = x_train
                    target = y_train
                startvalidate = time.time()         
                with torch.no_grad():
                    out1=net(inputs)
                endvalidate = time.time()
                predict_i = out1
                predict_i = torch.argmax(predict_i.reshape((6,predict_i.shape[2],predict_i.shape[3],predict_i.shape[4])),axis=0)
                # print(predict_i.shape)
                # new = np.copy(predict_i)
                dice_array = []
                for i in range(1,6):
                    lobe_predict_i = (predict_i == i)
                    lobe_label_i = (target == i)
                    dice = calculate_dice(lobe_predict_i,lobe_label_i)
                    stat[str(i)].append(dice.item())
                    #asd = calculate_ASD(lobe_predict_i,lobe_label_i,sampling = Spacing[::-1])
                    dice_array.append(round(dice.item(),4))
                    #asd_array.append(round(asd,4))
                    val_logger.info('dice {}:{} ({})'.format(i,round(dice.item(),4),val_count))
                avg.append(np.mean(np.array(dice_array)))
                # loss = criterion(out1, target)
                # val_loss += loss.item()
                valtime+=endvalidate-startvalidate
                del inputs,target,out1
            validate_loss=round(np.mean(np.array(avg)),4)
            val_logger.info('Validation loss: {} valTime: {}'.format(validate_loss,str(valtime/len(validation_loader))))
            val_logger.info('rm: dice:{} std:{}'.format(round(np.mean(np.array(rm)),4),round(np.std(np.array(rm)),4)))
            val_logger.info('ru: dice:{} std:{}'.format(round(np.mean(np.array(ru)),4),round(np.std(np.array(ru)),4)))
            val_logger.info('rl: dice:{} std:{}'.format(round(np.mean(np.array(rl)),4),round(np.std(np.array(rl)),4)))
            val_logger.info('lu: dice:{} std:{}'.format(round(np.mean(np.array(lu)),4),round(np.std(np.array(lu)),4)))
            val_logger.info('ll: dice:{} std:{}'.format(round(np.mean(np.array(ll)),4),round(np.std(np.array(ll)),4)))
            val_logger.info('avg: dice:{} std:{}'.format(round(np.mean(np.array(avg)),4),round(np.std(np.array(avg)),4)))
                
            if validate_loss>best_loss:
                best_loss=validate_loss
                save_cp=True
                Save_flag=0
            else:
                Save_flag+=1
                save_cp=False
                val_logger.info('Validation  loss  didnt improve from '+str(best_loss))
        if save_cp: 
            # Save checkpoint
            checkpoint = {
                'model': net.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            store_3 = 'loss_{:.4f}_CP{}.pth'.format(best_loss,epoch + 1)
            store_dir = os.path.join(cpt_store_path,store_3)
            if epoch + 1 < epochs:
                rmlst = glob.glob(os.path.join(cpt_store_path,'*.pth'))
                for item in rmlst:
                    os.remove(item)
            torch.save(checkpoint,store_dir)
            val_logger.info('Checkpoint {} saved !'.format(epoch + 1))
        if (epoch+1)%save_interval==0:
            checkpoint = {
                'model': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch':epoch
            }
            # store_3 = 'loss_{:.4f}_CP{}.pth'.format(best_loss,epoch + 1)
            store_dir = os.path.join(store_path,'save.pth')
            torch.save(checkpoint,store_dir)
            val_logger.info('Checkpoint for save {} saved !'.format(epoch + 1))
            
    train_logger.info('Training Completed')