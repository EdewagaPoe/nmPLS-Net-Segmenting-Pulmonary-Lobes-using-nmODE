from tqdm import tqdm
import os,json
import SimpleITK as sitk
import torch
import numpy as np
import logging
from scipy import ndimage as ndi
from scipy.ndimage import morphology
import torch.nn.functional as F

def get_precision(SR, GT, threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)
    TP = ((SR == 1) & (GT == 1))
    FP = ((SR == 1) & (GT == 0))
    PC = torch.sum(TP) / (torch.sum(TP + FP) + 1e-6)
    return PC
def get_recall(SR, GT, threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)
    TP = ((SR == 1) & (GT == 1))
    FN = ((SR == 0) & (GT == 1))
    RC = torch.sum(TP) / (torch.sum(TP + FN) + 1e-6)
    return RC
def get_specify(SR, GT, threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)
    TN = ((SR == 0) & (GT == 0))
    FP = ((SR == 1) & (GT == 0))
    RC = torch.sum(TN) / (torch.sum(TN + FP) + 1e-6)
    return RC

def get_accuracy(SR, GT, threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)
    TN = ((SR == 0) & (GT == 0))
    TP = ((SR == 1) & (GT == 1))
    FP = ((SR == 1) & (GT == 0))
    FN = ((SR == 0) & (GT == 1))
    RC = torch.sum(TN+TP) / (torch.sum(TN+TP+FN+FP) + 1e-6)
    return RC

def calculate_jaccard(SR,GT,threshold = 0.5):
    pred = SR > threshold
    gt = GT == torch.max(GT)
    intersection = torch.logical_and(pred, gt)
    union = torch.logical_or(pred, gt)
    iou = torch.sum(intersection) / (torch.sum(union)+1e-6)
    return iou

def calculate_dice(predict,label,smooth = 1):
    
    return (2*torch.sum(predict*label)+smooth)/(torch.sum(predict)+torch.sum(label)+smooth)

def chamfer_distance_transform(tensor):
    # Convert the binary tensor to a distance transform
    distance_transform = F.conv3d(tensor.unsqueeze(0).unsqueeze(0).float(), torch.ones(1, 1, 3, 3, 3).cuda(), padding=1)
    return distance_transform.squeeze()

def average_symmetric_surface_distance(tensor1, tensor2):
    # Convert CUDA tensors to PyTorch tensors
    tensor1 = tensor1.to(torch.float32)
    tensor2 = tensor2.to(torch.float32)

    # Convert binary tensors to distance transforms
    distance_transform1 = chamfer_distance_transform(tensor1)
    distance_transform2 = chamfer_distance_transform(tensor2)

    # Calculate the absolute difference between distance transforms
    abs_diff = torch.abs(distance_transform1 - distance_transform2)

    # Compute the ASSD as the mean of the absolute differences
    assd = abs_diff.mean().item()  # Convert the result to a Python float

    return assd

# def calculate_ASD(input1, input2, sampling=1, connectivity=1):
    
#     input_1 = np.atleast_1d(input1.astype(np.bool))
#     input_2 = np.atleast_1d(input2.astype(np.bool))
    

#     conn = morphology.generate_binary_structure(input_1.ndim, connectivity)

#     S = np.logical_xor(input_1 , morphology.binary_erosion(input_1, conn))
#     Sprime = np.logical_xor(input_2 , morphology.binary_erosion(input_2, conn))

    
#     dta = morphology.distance_transform_edt(~S,sampling)
#     dtb = morphology.distance_transform_edt(~Sprime,sampling)
    
#     sds = np.concatenate([np.ravel(dta[Sprime!=0]), np.ravel(dtb[S!=0])])
       
    
#     return sds.mean()
from scipy.ndimage import distance_transform_edt, binary_erosion
from scipy.ndimage.morphology import generate_binary_structure

def calculate_ASD(input1, input2):
    # Ensure inputs are boolean arrays
    input_1 = input1.astype(np.bool)
    input_2 = input2.astype(np.bool)

    # Calculate the symmetric surface by performing morphological operations
    S = input_1 ^ distance_transform_edt(input_1, return_distances=False, return_indices=True)[0]
    Sprime = input_2 ^ distance_transform_edt(input_2, return_distances=False, return_indices=True)[0]

    # Calculate the distance transforms
    dta = distance_transform_edt(~S, sampling=None)
    dtb = distance_transform_edt(~Sprime, sampling=None)

    # Calculate the distances for non-zero values efficiently
    sds = np.concatenate([dta[Sprime != 0], dtb[S != 0]])

    # Compute the mean of the distances
    assd = sds.mean()

    return assd

def calculate_ASD_GPU(input1, input2, sampling=1, connectivity=1):
    # Ensure inputs are CUDA tensors
    input_1 = input1.to(torch.bool)
    input_2 = input2.to(torch.bool)

    # Convert CUDA tensors to NumPy arrays
    input_1_np = input_1.cpu().numpy()
    input_2_np = input_2.cpu().numpy()

    # Define the binary structure for morphology operations
    conn = generate_binary_structure(input_1.ndim, connectivity)

    # Calculate the symmetric surface by performing morphological operations
    S = input_1_np ^ binary_erosion(input_1_np, conn)
    Sprime = input_2_np ^ binary_erosion(input_2_np, conn)

    # Calculate the distance transforms using SciPy
    dta = distance_transform_edt(~S, sampling)
    dtb = distance_transform_edt(~Sprime, sampling)

    # Calculate the distances for non-zero values efficiently
    sds = np.concatenate([dta[Sprime != 0], dtb[S != 0]])

    # Compute the mean of the distances
    assd = sds.mean()

    return assd
def dice_metric(predict,label,logger):
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
    new = np.copy(predict)
    
    
    dice_array = []
    for i in range(1,6):
        lobe_predict_i = (new == i)
        lobe_label_i = (label == i)
        dice = calculate_dice(lobe_predict_i,lobe_label_i)
        stat[str(i)].append(dice)
        #asd = calculate_ASD(lobe_predict_i,lobe_label_i,sampling = Spacing[::-1])
        dice_array.append(round(dice,4))
        #asd_array.append(round(asd,4))
    avg.append(np.mean(np.array(dice_array)))

    
    logger.info('rm: dice:{} std:{}'.format(round(np.mean(np.array(rm)),4),round(np.std(np.array(rm)),4)))
    logger.info('ru: dice:{} std:{}'.format(round(np.mean(np.array(ru)),4),round(np.std(np.array(ru)),4)))
    logger.info('rl: dice:{} std:{}'.format(round(np.mean(np.array(rl)),4),round(np.std(np.array(rl)),4)))
    logger.info('lu: dice:{} std:{}'.format(round(np.mean(np.array(lu)),4),round(np.std(np.array(lu)),4)))
    logger.info('ll: dice:{} std:{}'.format(round(np.mean(np.array(ll)),4),round(np.std(np.array(ll)),4)))
    logger.info('avg: dice:{} std:{}'.format(round(np.mean(np.array(avg)),4),round(np.std(np.array(avg)),4)))
    
    return round(np.mean(np.array(avg)),4)
