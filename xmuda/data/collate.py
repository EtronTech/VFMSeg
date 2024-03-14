import torch
from functools import partial


def collate_scn_base(input_dict_list, output_orig, with_vfm,output_image=True):
    """
    Custom collate function for SCN. The batch size is always 1,
    but the batch indices are appended to the locations.
    :param input_dict_list: a list of dicts from the dataloader
    :param output_orig: whether to output original point cloud/labels/indices
    :param output_image: whether to output images
    :return: Collated data batch as dict
    """
    locs=[]
    feats=[]
    labels=[]

    if output_image:
        imgs = []
        img_idxs = []

        if with_vfm:
            img_paths = [] # Output Image Path
            # img_instances = [] # Output Image Instances
            img_indices_orig = [] # Output Image Indices
            img_instances_orig = [] # Output Original Image


    if output_orig:
        orig_seg_label = []
        orig_points_idx = []

    output_pselab = 'pseudo_label_2d' in input_dict_list[0].keys()
    if output_pselab:
        pseudo_label_2d = []
        pseudo_label_3d = []

    # print('collate BatchSz: ', len(input_dict_list))

    for idx, input_dict in enumerate(input_dict_list):
        coords = torch.from_numpy(input_dict['coords'])
        batch_idxs = torch.LongTensor(coords.shape[0], 1).fill_(idx)
        locs.append(torch.cat([coords, batch_idxs], 1))
        feats.append(torch.from_numpy(input_dict['feats']))
        if 'seg_label' in input_dict.keys():
            labels.append(torch.from_numpy(input_dict['seg_label']))
        if output_image:
            imgs.append(torch.from_numpy(input_dict['img']))
            img_idxs.append(input_dict['img_indices'])
        if output_orig:
            orig_seg_label.append(input_dict['orig_seg_label'])
            orig_points_idx.append(input_dict['orig_points_idx'])
        if output_pselab:
            pseudo_label_2d.append(torch.from_numpy(input_dict['pseudo_label_2d']))
            if input_dict['pseudo_label_3d'] is not None:
                pseudo_label_3d.append(torch.from_numpy(input_dict['pseudo_label_3d']))
        
        if with_vfm:
            img_paths.append(input_dict['img_paths']) # Output Image Path
            # img_instances.append(torch.from_numpy(input_dict['img'])) # Output Image Instances
            img_instances_orig.append(torch.from_numpy(input_dict['img_instances_orig'])) # Output Original Image Instances
            img_indices_orig.append(input_dict['img_indices_orig']) # Output Image Original Indices

    locs = torch.cat(locs, 0)
    feats = torch.cat(feats, 0)
    out_dict = {'x': [locs, feats]}
    if labels:
        labels = torch.cat(labels, 0)
        out_dict['seg_label'] = labels
    if output_image:
        out_dict['img'] = torch.stack(imgs)
        out_dict['img_indices'] = img_idxs
        
        if with_vfm:
            out_dict['img_paths'] = img_paths # Output Image Path
            # out_dict['img_instances'] = img_instances # Output Batch Image Instance
            out_dict['img_indices_orig'] = img_indices_orig # Output Image  Original image Indices
            out_dict['img_instances_orig'] = img_instances_orig # Output Batch Original Image Instance
        
    if output_orig:
        out_dict['orig_seg_label'] = orig_seg_label
        out_dict['orig_points_idx'] = orig_points_idx
    if output_pselab:
        out_dict['pseudo_label_2d'] = torch.cat(pseudo_label_2d, 0)
        out_dict['pseudo_label_3d'] = torch.cat(pseudo_label_3d, 0) if pseudo_label_3d else pseudo_label_3d
    return out_dict


def get_collate_scn(is_train,with_vfm):
    return partial(collate_scn_base,
                   output_orig=not is_train,
                   with_vfm=with_vfm,
                   )
