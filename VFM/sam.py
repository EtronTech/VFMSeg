import cv2
import torch
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import time

from torchvision import transforms

def build_SAM(sam_ckpt=None,device=0):
    if sam_ckpt == None:
        path = '/Labs/Scripts/3DPC/xMUDA/VFM_ckpt/'
        sam_ckpt = path + "sam_vit_h_4b8939.pth"
    
    model_type = "vit_h"

    sam = sam_model_registry[model_type](checkpoint=sam_ckpt)
    sam.cuda(device)

    mask_generator = SamAutomaticMaskGenerator(sam)

    return mask_generator


def discard(set_data,val=0):
    set_data.discard(val) # Mask ID starts from 1
    return set_data

def sample(ids_list):#(ids_lists_src,ids_lists_trg):
    return np.random.choice(ids_list,len(ids_list)//2,replace=False)

def generate_mixed_data_with_SAM(src_sam_masks_data,trg_sam_masks_data,
                                  src_img_indices,
                                  src_2d_data,src_3d_data,
                                  src_seg_labels,
                                  trg_img_indices,
                                  trg_2d_data,trg_3d_data,
                                  trg_2d_pl,
                                  trg_3d_pl,
                                  device=0):
    src_2d_with_trg_mix = []
    trg_2d_with_src_mix = []
    src_2d_with_trg_mix_label = []
    trg_2d_with_src_mix_label = []
    src_3d_with_trg_mix = []
    trg_3d_with_src_mix = []
    mixed_batch = {}

    random_mix = {}
    cut_mix = {}
    
    end = time.time() 
    # Iterate SAM masks, find all Ids, flatten, sample 50% masks
    # (Batch,src/trg),  src[Mask_Ids] trg[Mask_Ids]
    ids_lists = [ sample(list(discard(set((src_masks).numpy().flatten()))),list(discard(set((trg_masks).numpy().flatten())))) for src_masks,trg_masks in zip(src_sam_masks_data,trg_sam_masks_data)]
    data_time1 = time.time() - end
    
    end = time.time()
    for batch_id in range(len(src_sam_masks_data)):
        # Restore SAM Masks from sampled Mask IDs
        src_mask = (src_sam_masks_data[batch_id] == ids_lists[batch_id][0][0]).numpy() # save first mask
        for id in range(1,len(ids_lists[batch_id][0])):
            src_mask = src_mask | (src_sam_masks_data[batch_id] == ids_lists[batch_id][0][id]).numpy() # Merge Sampled Masks
        trg_mask = (trg_sam_masks_data[batch_id] == ids_lists[batch_id][1][0]).numpy() # save first mask
        for id in range(1,len(ids_lists[batch_id][1])):
            trg_mask = trg_mask | (trg_sam_masks_data[batch_id] == ids_lists[batch_id][1][id]).numpy() # Merge Sampled Masks
        
        # Create Image Label Mix (batch label concat 0 dim)
        src_img_indices
        src_2d_with_trg_mix_label
        trg_2d_with_src_mix_label

        # Create Image Mix
        src_mask = torch.from_numpy(src_mask).unsqueeze(0).repeat(3,1,1)
        trg_mask = torch.from_numpy(trg_mask).unsqueeze(0).repeat(3,1,1)

        src_mix = src_2d_data[batch_id].clone()
        src_mix[trg_mask] = trg_2d_data[batch_id][trg_mask]
        src_2d_with_trg_mix.append(src_mix)

        trg_mix = trg_2d_data[batch_id].clone()
        trg_mix[src_mask] = src_2d_data[batch_id][src_mask]
        trg_2d_with_src_mix.append(trg_mix)

    data_time3 = time.time() - end


    return mixed_batch, random_mix, cut_mix