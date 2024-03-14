import torch
from functools import partial

import time


def collate_scn_base(input_dict_list, output_orig, with_vfm,output_image=True):
    """
    Custom collate function for SCN. The batch size is always 1,
    but the batch indices are appended to the locations.
    :param input_dict_list: a list of dicts from the dataloader
    :param output_orig: whether to output original point cloud/labels/indices
    :param output_image: whether to output images
    :return: Collated data batch as dict
    """

    labels=[]

    if output_image:

        img_idxs = []

        if with_vfm:
            # img_paths = [] # Output Image Path
            # # img_instances = [] # Output Image Instances
            # img_indices_orig = [] # Output Image Indices
            # img_instances_orig = [] # Output Original Image


            sampled_sam_mask = []
            sam_mix_indices = []
            sam_mix_image = []
            sam_mix_label_2d = []
            sam_label = []

            cut_mask = []
            cut_mix_indices = []
            cut_mix_image = []
            cut_mix_label_2d = []
            cut_label = []


    output_pselab = 'pseudo_label_2d' in input_dict_list[0].keys()
    if output_pselab:
        pseudo_label_2d = []
        pseudo_label_3d = []

        sam_mix_pseudo_label_2d = []
        sam_pseudo_label_2d = []

        cut_mix_pseudo_label_2d = []
        cut_pseudo_label_2d = []
    
    sam_seg_labels = 'sam_mix_label_2d' in input_dict_list[0].keys()
    exist_sam_mix_indices = 'sam_mix_indices' in input_dict_list[0].keys()
    

    # print('collate BatchSz: ', len(input_dict_list))

    for idx, input_dict in enumerate(input_dict_list):

        if 'seg_label' in input_dict.keys():
            # labels.append(torch.from_numpy(input_dict['seg_label']))
            labels.append(input_dict['seg_label']) 
            if not output_pselab and sam_seg_labels:
                sam_mix_label_2d.append(input_dict['sam_mix_label_2d'])
                sam_label.append(input_dict['sam_label'])
                cut_mix_label_2d.append(input_dict['cut_mix_label_2d'])
                cut_label.append(input_dict['cut_label'])

        if output_image:
            img_idxs.append(input_dict['img_indices'])
            

            if exist_sam_mix_indices:
                sam_mix_indices.append(input_dict['sam_mix_indices'])
                cut_mix_indices.append(input_dict['cut_mix_indices'])
            
        if output_pselab:
            # pseudo_label_2d.append(torch.from_numpy(input_dict['pseudo_label_2d']))
            pseudo_label_2d.append(input_dict['pseudo_label_2d'])
            sam_mix_pseudo_label_2d.append(input_dict['sam_mix_pseudo_label_2d'])
            cut_mix_pseudo_label_2d.append(input_dict['cut_mix_pseudo_label_2d'])
            sam_pseudo_label_2d.append(input_dict['sam_pseudo_label_2d'])
            cut_pseudo_label_2d.append(input_dict['cut_pseudo_label_2d'])
            
        
        if with_vfm:
            # img_paths.append(input_dict['img_paths']) # Output Image Path
            # # img_instances.append(torch.from_numpy(input_dict['img'])) # Output Image Instances
            # img_instances_orig.append(torch.from_numpy(input_dict['img_instances_orig'])) # Output Original Image Instances
            # img_indices_orig.append(input_dict['img_indices_orig']) # Output Image Original Indices

            sampled_sam_mask.append(input_dict['sampled_sam_mask'])
            cut_mask.append(input_dict['cut_mask'])
            


    out_dict = {}
    if labels:
        
        if not output_pselab and sam_seg_labels:
            # Target data， Mixed Data using pseudo labels
            out_dict['sam_mix_label_2d'] = sam_mix_label_2d
            # out_dict['sam_mix_label_2d_indices'] = 
            out_dict['cut_mix_label_2d'] = cut_mix_label_2d
            out_dict['sam_label'] = sam_label
            out_dict['cut_label'] = cut_label
                                                  
    if output_image:
        out_dict['img_indices'] = img_idxs


    if with_vfm:
        out_dict['sampled_sam_mask'] = torch.stack(sampled_sam_mask)
        out_dict['cut_mask'] = torch.stack(cut_mask)


    if output_pselab:
        out_dict['pseudo_label_2d'] = torch.cat(pseudo_label_2d, 0)
        
        out_dict['sam_mix_pseudo_label_2d'] = sam_mix_pseudo_label_2d #torch.cat(sam_mix_pseudo_label_2d, 0)
        out_dict['cut_mix_pseudo_label_2d'] = cut_mix_pseudo_label_2d #torch.cat(cut_mix_pseudo_label_2d, 0)     
        out_dict['sam_pseudo_label_2d'] = sam_pseudo_label_2d #torch.cat(sam_mix_pseudo_label_2d, 0)
        out_dict['cut_pseudo_label_2d'] = cut_pseudo_label_2d #torch.cat(cut_mix_pseudo_label_2d, 0)    

    return out_dict


def get_collate_scn(is_train,with_vfm):
    return partial(collate_scn_base,
                   output_orig=not is_train,
                   with_vfm=with_vfm,
                   )
