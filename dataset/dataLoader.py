import os
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import pickle
import numpy as np
import os,random,glob
import SimpleITK as sitk
from torchvision import transforms as tfs
from skimage.morphology import closing,binary_dilation
from skimage.transform import resize,downscale_local_mean,rescale
import json
from scipy.ndimage import zoom
from tqdm.notebook import trange, tqdm 
import logging
def readimage(filedir):
    return np.load(filedir)

def readmhd(filedir):
    itkimage = sitk.ReadImage(filedir)
    return sitk.GetArrayFromImage(itkimage)


def readmhd_spaceing(filedir):
    itkimage = sitk.ReadImage(filedir)
    Spacing = itkimage.GetSpacing()
    return sitk.GetArrayFromImage(itkimage),Spacing

def rescale_ww_wl(data,window_width = 1600,window_center = -400):
    
    
    min_ = (2*window_center - window_width)/2.0 + 0.5;  
    max_ = (2*window_center + window_width)/2.0 + 0.5;
    
    res = (data - min_)*255.0/(max_ - min_)
    res[res<0] = 0
    res[res>255] = 255
    
    return res

def rescale_max_min(data,max_,min_):
    
    data[data>max_] = 1
    data[data<min_] = 0
    data[np.where((data<=max_)&(data>=min_))] = (data[np.where((data<=max_)&(data>=min_))] - min_)/(max_-min_)   
    
    return data


def scaler(volumes):
    if np.max(volumes)>0:
        return (volumes - np.min(volumes)) / (np.max(volumes) - np.min(volumes))
    else:
        return volumes


def dice_coef(input_data, target, smooth=1):

    iflat = input_data.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    
    return (2. * intersection.item() + smooth) /(iflat.sum().item() + tflat.sum().item() + smooth)

def read_data(datapath, series_id):
    reader = sitk.ImageSeriesReader()
    dcm_names = reader.GetGDCMSeriesFileNames(datapath, series_id)
    reader.SetFileNames(dcm_names)
    itk_img = reader.Execute()
    spacing = itk_img.GetSpacing()
    origin = itk_img.GetOrigin()
    img_arr = sitk.GetArrayFromImage(itk_img)
    return img_arr, spacing[::-1], origin
def trans(data, hu_bound=[-1000, 400]):
    data = (data - hu_bound[0]) / (hu_bound[1] - hu_bound[0])
    data[data<0] = 0.
    data[data>1] = 1.
    # return torch.from_numpy(data).float()
    return torch.from_numpy(data[np.newaxis, ...]).float()

def resample(image, spacing, new_spacing=[1,1,1]):
    spacing = np.array(list(spacing))
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    image = zoom(image, real_resize_factor, mode='nearest')
    
    return image

path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'datafile.json')# take place datafile.json here to your own file 
with open(path) as file_obj:
    dataset = json.load(file_obj)


class Fissure_Pertrain_Dataset(Dataset):
    def __init__(self,
                 start=0,
                 end = 100,
                 crop_size = 20,
                 slice_crop=20,
                 data_root='',
                 data_list_file=''):
        self.crop_size = crop_size
        self.slice_crop = slice_crop
        self.data_list = np.load(data_list_file,allow_pickle=True)[start:end]      
        local_path = data_root
        
        self.seriesList = self.data_list
    def __len__(self):
        return len(self.seriesList)
    def __getitem__(self, index):
        data = self.seriesList[index]
        path = data['path']
        sid = data['sid']
        cropsize = self.crop_size
        slicecrop = self.slice_crop
        img_arr,spacing,origin = read_data(path,sid)
        img_arr = resample(img_arr,spacing)
        img_arr = img_arr[slicecrop:-slicecrop,cropsize:-cropsize,cropsize:-cropsize]
        img_arr = trans(img_arr)
        return img_arr

class Lobe_Dataset_multitask_20200318(Dataset):
    
    def __init__(self,data_set_type='',dataroot='',crop_size = 20,
                 slice_crop=20,):
        self.data_dirs = dataset[data_set_type]
        #self.data_dirs = None
        self.crop_size = crop_size
        self.slice_crop = slice_crop
        self.dataroot = dataroot
    def __len__(self):
        
        return len(self.data_dirs)

    def __getitem__(self, index):
        
        i=index%len(self.data_dirs)
        
        lung_file_name = os.path.basename(self.data_dirs[i])
        print('filename:',self.data_dirs[i])
        lung_dir =  self.dataroot+self.data_dirs[i]+'/'+lung_file_name+'.mha'
        lobe_dir =  self.dataroot+self.data_dirs[i] +'/lobe.mha'
        border_dir = self.dataroot+self.data_dirs[i] +'/border.mha'
        
        if os.path.exists(lobe_dir) is not True:
            lobe_dir =  self.dataroot+self.data_dirs[i] +'/lobe_mask.mha'
        
        if os.path.exists(lung_dir) and os.path.exists(lobe_dir) and os.path.exists(border_dir):
            volumes=readmhd(lung_dir)      
            # volumes = trans(volumes)
            #volumes = rescale_ww_wl(volumes)
            labels = readmhd(lobe_dir)
            border = readmhd(border_dir)
            border = (border > 0.5)
            #crop the x*400*400 volumes
            # x1_begin = int((volumes.shape[0]-self.crop_size[0])/2)
            crop_volumes = volumes[self.slice_crop:-self.slice_crop,self.crop_size:-self.crop_size,self.crop_size:-self.crop_size]
            crop_volumes = trans(crop_volumes)
            # crop_volumes = crop_volumes.reshape((1,crop_volumes.shape[0],crop_volumes.shape[1],crop_volumes.shape[2]))
            crop_labels = labels[self.slice_crop:-self.slice_crop,self.crop_size:-self.crop_size,self.crop_size:-self.crop_size]
            #crop_labels = crop_labels.reshape((1,crop_labels.shape[0],crop_labels.shape[1],crop_labels.shape[2]))
            crop_border = border[self.slice_crop:-self.slice_crop,self.crop_size:-self.crop_size,self.crop_size:-self.crop_size]
            crop_border = crop_border.reshape((1,crop_border.shape[0],crop_border.shape[1],crop_border.shape[2]))
            return crop_volumes,crop_labels,crop_border
        
class Lobe_Dataset_Resampled(Dataset):
    
    def __init__(self,data_set_type='',dataroot='',crop_size:int = 40,
                 slice_crop=40,maskout:bool = False,logger=None,border = False):
       
    
        self.data_dirs = dataset[data_set_type]
        #self.data_dirs = None
        self.crop_size = crop_size
        self.slice_crop = slice_crop
        self.dataroot = dataroot
        self.maskout = maskout
        self.logger = logger
        self.border = border
        print('maskout',maskout)
    def __len__(self):
        
        return len(self.data_dirs)

    def __getitem__(self, index):
        
        i=index%len(self.data_dirs)
        path = os.path.join(self.dataroot,os.path.basename(self.data_dirs[i]))
        lung_file_name = os.path.basename(self.data_dirs[i])
        if self.logger!= None:
            self.logger.info('filename:'+os.path.basename(self.data_dirs[i]))
        else:
            print('filename:'+os.path.basename(self.data_dirs[i]))
        lung_dir =  os.path.join(path,'lung.mha')
        lobe_dir =  os.path.join(path,'mask.mha')
        border_dir = os.path.join(path,'generated_border.mha')
        # lobe_dir =  os.path.join(path,'filled_mask.mha')  
        
        if os.path.exists(lobe_dir) is not True:
            lobe_dir =  self.dataroot+self.data_dirs[i] +'/lobe_mask.mha'
        
        if os.path.exists(lung_dir) and os.path.exists(lobe_dir) and os.path.exists(border_dir) :
            volumes=readmhd(lung_dir)      
            labels = readmhd(lobe_dir)
            self.crop_size = self.crop_size if volumes.shape[1]>350 else 1
            crop_volumes = volumes[self.slice_crop:-self.slice_crop, self.crop_size:-self.crop_size, self.crop_size:-self.crop_size]
            crop_labels = labels[self.slice_crop:-self.slice_crop,self.crop_size:-self.crop_size,self.crop_size:-self.crop_size]
            crop_volumes = trans(crop_volumes)
            if self.maskout:
                mask = (crop_labels==0)
                mask = torch.from_numpy(mask[np.newaxis, ...])
                crop_volumes[mask]=-1
            if self.border:
                border = readmhd(border_dir)
                crop_border = border[self.slice_crop:-self.slice_crop,self.crop_size:-self.crop_size,self.crop_size:-self.crop_size]
                return crop_volumes,crop_labels,crop_border
            return crop_volumes,crop_labels

class Luna16_51(Dataset):
    
    def __init__(self,dataset='',dataset_type='train',datafile='',dataroot='',crop_size:int = 40,
                 slice_crop=20,maskout:bool = False,logger=None,border = False):
       
    
        with open(datafile,'r') as f:
            datasets = json.load(f)
        #self.data_dirs = None
        self.data_dirs = datasets[dataset][dataset_type]
        self.crop_size = crop_size
        self.slice_crop = slice_crop
        self.dataroot = dataroot
        self.maskout = maskout
        self.logger = logger
        self.border = border
        # print('maskout',maskout)
    def __len__(self):
        
        return len(self.data_dirs)

    def __getitem__(self, index):
        
        i=index%len(self.data_dirs)
        path = os.path.join(self.dataroot,os.path.basename(self.data_dirs[i]))
        lung_file_name = os.path.basename(self.data_dirs[i])
        if self.logger!= None:
            self.logger.info('filename:'+os.path.basename(self.data_dirs[i]))
        else:
            print('filename:'+os.path.basename(self.data_dirs[i]))
        lung_dir =  os.path.join(path,'lung.mha')
        lobe_dir =  os.path.join(path,'mask.mha')
        # border_dir = os.path.join(path,'generated_border.mha')
        # lobe_dir =  os.path.join(path,'filled_mask.mha')  
        
        if os.path.exists(lobe_dir) is not True:
            lobe_dir =  self.dataroot+self.data_dirs[i] +'/lobe_mask.mha'
        
        if os.path.exists(lung_dir) and os.path.exists(lobe_dir): #and os.path.exists(border_dir) :
            volumes=readmhd(lung_dir)      
            labels = readmhd(lobe_dir)
            size = labels.shape[1]
            crop_size = max(int(size/10),self.crop_size)
            self.crop_size = crop_size if volumes.shape[1]>300 else 1
            crop_volumes = volumes[self.slice_crop:-self.slice_crop, self.crop_size:-self.crop_size, self.crop_size:-self.crop_size]
            crop_labels = labels[self.slice_crop:-self.slice_crop,self.crop_size:-self.crop_size,self.crop_size:-self.crop_size]
            crop_labels = crop_labels.astype('int8')
            crop_volumes = trans(crop_volumes)
            if self.maskout:
                mask = (crop_labels==0)
                mask = torch.from_numpy(mask[np.newaxis, ...])
                crop_volumes[mask]=-1
            # if self.border:
            #     border = readmhd(border_dir)
            #     crop_border = border[self.slice_crop:-self.slice_crop,self.crop_size:-self.crop_size,self.crop_size:-self.crop_size]
            #     return crop_volumes,crop_labels,crop_border
            print(crop_volumes.shape,crop_labels.shape,crop_size)
            return crop_volumes,crop_labels
