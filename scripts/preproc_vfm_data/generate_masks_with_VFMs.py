import torch
import cv2
import numpy as np
import os.path as osp
import pickle
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
import argparse
from tqdm import tqdm
import time
import os

import sys
sys.path.append('/Labs/Scripts/3DPC/VFMSeg')
sys.path.append('/Labs/Scripts/3DPC/VFMSeg/VFM')
from xmuda.data.utils.augmentation_3d import augment_and_scale_3d
from VFM.sam import build_SAM
from VFM.seem import build_SEEM, call_SEEM

from torch.utils.data.sampler import RandomSampler, BatchSampler
from torch.utils.data.dataloader import DataLoader, default_collate
from yacs.config import CfgNode as CN



# prevent "RuntimeError: received 0 items of ancdata"
torch.multiprocessing.set_sharing_strategy('file_system')


cuda_device_idx = 0
# torch.cuda.set_device(cuda_device_idx)


# Settings
pre_defined_proc_list =  ['A2D2_Resize'] #['SKITTI_A2D2'] #['DAY_NIGHT_Resize'] #['USA_SING_Resize'] # ['A2D2_Resize']#[ 'USA_SING' ,  'DAY_NIGHT'] # 'KITTI', 'A2D2' , 'USA_SING' ,  'DAY_NIGHT'

# Path for VFM ckpt and config files:
vfm_pth='/Labs/Scripts/3DPC/xMUDA/VFM_ckpt/seem_focall_v1.pt'
vfm_cfg='../../VFM/configs/seem/seem_focall_lang.yaml'

#######################
# Config
######################

#  Virtual_SemantiKITTI
vkitti_orig_data_path = '/Labs/Scripts/3DPC/Datasets/3DPC/virtual_kitti'
save_new_vkitti_pkl_path = '/Labs/Scripts/3DPC/Datasets/3DPC/XMUDA_WITH_VFM/virtual_kitti_preprocess_vfm'
pkl_path_VKITTI_train = '/Labs/Scripts/3DPC/Datasets/3DPC/nuScenes_preprocess/virtual_kitti_preprocess/preprocess/train.pkl'

save_new_skitti_pkl_path = '/Labs/Scripts/3DPC/Datasets/3DPC/XMUDA_WITH_VFM/semantic_kitti_preprocess_vfm'
skitti_orig_data_path = '/Labs/Scripts/3DPC/Datasets/3DPC/SKITTI/semantic_kitti'
pkl_path_SKITTI_train = '/Labs/Scripts/3DPC/Datasets/3DPC/nuScenes_preprocess/semantickitti-preprocess/preprocess/train.pkl'
pkl_path_SKITTI_test   = '/Labs/Scripts/3DPC/Datasets/3DPC/nuScenes_preprocess/semantickitti-preprocess/preprocess/test.pkl'
pkl_path_SKITTI_val  = '/Labs/Scripts/3DPC/Datasets/3DPC/nuScenes_preprocess/semantickitti-preprocess/preprocess/val.pkl'
config_path_VKITTI = '/Labs/Scripts/3DPC/VFMSeg/configs/virtual_kitti_semantic_kitti/uda/xmuda_pl.yaml'

#  A2D2_SemanticKITTI
a2d2_orig_preprocess_data_path = '/Labs/Scripts/3DPC/Datasets/3DPC/nuScenes_preprocess/audi/a2d2_preprocess'
save_new_a2d2_pkl_path = '/Labs/Scripts/3DPC/Datasets/3DPC/XMUDA_WITH_VFM/a2d2_preprocess_vfm'
pkl_path_A2D2_train = '/Labs/Scripts/3DPC/Datasets/3DPC/nuScenes_preprocess/audi/preprocess/train.pkl'


# nuScene lidarseg
nuScene_orig_data_path = '/Labs/Scripts/3DPC/Datasets/3DPC/nuScenes'

save_new_usa_singapore_pkl_path = '/Labs/Scripts/3DPC/Datasets/3DPC/XMUDA_WITH_VFM/nuscene_preprocess_vfm/usa_singapore'
pkl_path_train_usa_path = '/Labs/Scripts/3DPC/Datasets/3DPC/nuScenes_preprocess/nuScenes_lidarseg/preprocess/train_usa.pkl'
pkl_path_train_singapore_path = '/Labs/Scripts/3DPC/Datasets/3DPC/nuScenes_preprocess/nuScenes_lidarseg/preprocess/train_singapore.pkl'
pkl_path_test_usa_path = '/Labs/Scripts/3DPC/Datasets/3DPC/nuScenes_preprocess/nuScenes_lidarseg/preprocess/test_usa.pkl'
pkl_path_test_singapore_path = '/Labs/Scripts/3DPC/Datasets/3DPC/nuScenes_preprocess/nuScenes_lidarseg/preprocess/test_singapore.pkl'
pkl_path_val_singapore_path = '/Labs/Scripts/3DPC/Datasets/3DPC/nuScenes_preprocess/nuScenes_lidarseg/preprocess/val_singapore.pkl'

save_new_day_night_pkl_path = '/Labs/Scripts/3DPC/Datasets/3DPC/XMUDA_WITH_VFM/nuscene_preprocess_vfm/day_night'
pkl_path_train_day_path = '/Labs/Scripts/3DPC/Datasets/3DPC/nuScenes_preprocess/nuScenes_lidarseg/preprocess/train_day.pkl'
pkl_path_train_night_path = '/Labs/Scripts/3DPC/Datasets/3DPC/nuScenes_preprocess/nuScenes_lidarseg/preprocess/train_night.pkl'
pkl_path_test_day_path = '/Labs/Scripts/3DPC/Datasets/3DPC/nuScenes_preprocess/nuScenes_lidarseg/preprocess/test_day.pkl'
pkl_path_test_night_path = '/Labs/Scripts/3DPC/Datasets/3DPC/nuScenes_preprocess/nuScenes_lidarseg/preprocess/test_night.pkl'
pkl_path_val_night_path = '/Labs/Scripts/3DPC/Datasets/3DPC/nuScenes_preprocess/nuScenes_lidarseg/preprocess/val_night.pkl'

######################

def merge_sam_masks(image_masks):
    num_of_masks = len(image_masks)
    merged_masks = []
    if num_of_masks > 0:
        merged_masks = np.zeros((image_masks[0].shape[0],image_masks[0].shape[1]),dtype=np.int16)
        for i in range(num_of_masks):
            merged_masks[image_masks[i]] = i+1 # mask index starts from 1
        return merged_masks
    else:
        return []
  

def preprocess_vkitti_train(SEEM,SAM):
    mapping = 'SemanticKITTISCN'
    curr_split = 'train'
    pkl_data = []
    # 1. Load Pickle Data
    # load VKITTI
    print('load pkl data...')
    with open(pkl_path_VKITTI_train,'rb') as f:
        pkl_data.extend(pickle.load(f))

    # data 2126 items, 
    # keys
    print('iterate pkl data...')
    save_dir = save_new_vkitti_pkl_path + '/' + curr_split
    os.makedirs(save_dir, exist_ok=True)
    
    random_weather =  ['clone', 'fog', 'morning', 'overcast', 'rain', 'sunset']
    # 2. Iterate PKL
    pkl_files = []
    accu_id = 0
    try:
        with tqdm(total=len(pkl_data)*6) as bar:
            for pkl_id, data in  enumerate(pkl_data):
                new_pkl = {}

                images = {}
                for weather in random_weather:
                    img_path = osp.join(vkitti_orig_data_path, 'vkitti_1.3.1_rgb', data['scene_id'], weather,
                                data['frame_id'] + '.png') 
                    images[weather]= Image.open(img_path)

                image_sam_masks = {} # Weather as Key
                image_seem_masks = {}# Weather as Key
        # 3. Generate Masks (6 Weather Images)
                for idx, key in enumerate(images):
                    sam_masks = SAM.generate(cv2.cvtColor(np.array(images[key]).astype(np.uint8), cv2.COLOR_BGR2RGB))
                    seem_masks,_ = call_SEEM(SEEM,pil_image=images[key],mapping=mapping)

                    sam_masks = [x['segmentation'] for x in sam_masks]   # [mask, h, w]
                    sam_masks = merge_sam_masks(sam_masks)  # Merge All mask into single 'Image'
                    seem_masks = [x for x in seem_masks.cpu().numpy()]

                    image_sam_masks[key] = sam_masks
                    image_seem_masks[key] = seem_masks

                    bar.update(1)

                new_pkl['sam'] = image_sam_masks.copy()
                new_pkl['seem'] = image_seem_masks.copy()

                pkl_files.append(new_pkl)

        # 4. Save PKL Data
                # save to pickle file
                if 0 == (len(pkl_files) % 5):
                    for pkl in pkl_files:
                        pkl_name = '{}.pkl'.format(str(accu_id))
                        with open(osp.join(save_dir, pkl_name), 'wb') as f:
                            pickle.dump([pkl], f)
                        accu_id += 1
                    pkl_files = []

        if 0 != len(pkl_files):
            for pkl in pkl_files:
                pkl_name = '{}.pkl'.format(str(accu_id))
                with open(osp.join(save_dir, pkl_name), 'wb') as f:
                    pickle.dump([pkl], f)
                accu_id += 1
            pkl_files = []
    
    except:
        print('Exception Occured!')
    else:
        pass

    return


def preprocess_skitti(SEEM,SAM,dataset='',mapping='SemanticKITTISCN'):

    mapping = mapping
    pkl_data = []

    curr_split = dataset
    # 1. Load Pickle Data
    print('load pkl data...')
    # data items, 
    # keys
    if 'train' == dataset:
        pkl_path = pkl_path_SKITTI_train
    elif 'val' == dataset:
        pkl_path = pkl_path_SKITTI_val
    elif 'test' == dataset:
        pkl_path = pkl_path_SKITTI_test
    else:
        return
    
    if mapping == 'SemanticKITTISCN':
        save_dir = save_new_skitti_pkl_path +'/' + curr_split
    else:
        save_dir = save_new_skitti_pkl_path +'/' + curr_split + '_for_a2d2'
    os.makedirs(save_dir, exist_ok=True)
    with open(pkl_path,'rb') as f:
        pkl_data.extend(pickle.load(f))
    
    # 2. Iterate PKL
    print('iterate pkl data...')
    pkl_files = []
    accu_id = 0
    with tqdm(total=len(pkl_data)) as bar:
        # train data 18029 items, 
        for pkl_id, data in  enumerate(pkl_data):
            new_pkl = {}
            img_path = osp.join(skitti_orig_data_path, data['camera_path'])
            image = Image.open(img_path)

    # 3. Generate Masks 
            sam_masks = SAM.generate(cv2.cvtColor(np.array(image).astype(np.uint8), cv2.COLOR_BGR2RGB))
            seem_masks,_ = call_SEEM(SEEM,pil_image=image,mapping=mapping)


            sam_masks = [x['segmentation'] for x in sam_masks]   # [mask, h, w]
            sam_masks = merge_sam_masks(sam_masks)  # Merge All mask into single 'Image'
            seem_masks =  [x for x in seem_masks.cpu().numpy()]

            new_pkl['sam'] = sam_masks.copy()
            new_pkl['seem'] = seem_masks.copy()

            pkl_files.append(new_pkl)

            if 0 == (len(pkl_files) % 10):
    # 4. Save PKL Data
    # save to pickle file
                for pkl in pkl_files:
                    pkl_name = '{}.pkl'.format(str(accu_id))
                    with open(osp.join(save_dir, pkl_name), 'wb') as f:
                        pickle.dump([pkl], f)
                    accu_id += 1
                pkl_files = []
            
            bar.update(1)
        
    if 0 != len(pkl_files):
        for pkl in pkl_files:
            pkl_name = '{}.pkl'.format(str(accu_id))
            with open(osp.join(save_dir, pkl_name), 'wb') as f:
                pickle.dump([pkl], f)
            accu_id += 1
        
    return




def preprcess_a2d2(SEEM,SAM,dataset='',Resize=None):
    mapping = 'A2D2SCN'
    pkl_data = []

    curr_split = dataset
    # 1. Load Pickle Data
    print('load pkl data...')
    # data items, 
    # keys
    if 'train' == dataset:
        pkl_path = pkl_path_A2D2_train
    else:
        return
    if Resize:
        save_dir = save_new_a2d2_pkl_path +'/' + curr_split + '_resize'
    else:
        save_dir = save_new_a2d2_pkl_path +'/' + curr_split
    os.makedirs(save_dir, exist_ok=True)
    with open(pkl_path,'rb') as f:
        pkl_data.extend(pickle.load(f))
    
    # 2. Iterate PKL
    print('iterate pkl data...')
    pkl_files = []
    accu_id = 0
    with tqdm(total=len(pkl_data)) as bar:
        # train data 27695 items, 
        for pkl_id, data in  enumerate(pkl_data):           
            new_pkl = {}
            img_path = osp.join(a2d2_orig_preprocess_data_path, data['camera_path'])
            image = Image.open(img_path)

            if not image.size == Resize:
                image = image.resize(Resize, Image.BILINEAR)
            else:
                print('Use Original Image')

    # 3. Generate Masks 
            sam_masks = SAM.generate(cv2.cvtColor(np.array(image).astype(np.uint8), cv2.COLOR_BGR2RGB))
            seem_masks,_ = call_SEEM(SEEM,pil_image=image,mapping=mapping)


            sam_masks = [x['segmentation'] for x in sam_masks]   # [mask, h, w]
            sam_masks = merge_sam_masks(sam_masks)  # Merge All mask into single 'Image'
            seem_masks =  [x for x in seem_masks.cpu().numpy()]

            new_pkl['sam'] = sam_masks.copy()
            new_pkl['seem'] = seem_masks.copy()

            pkl_files.append(new_pkl)

            if 0 == (len(pkl_files) % 10):
    # 4. Save PKL Data
    # save to pickle file
                for pkl in pkl_files:
                    pkl_name = '{}.pkl'.format(str(accu_id))
                    with open(osp.join(save_dir, pkl_name), 'wb') as f:
                        pickle.dump([pkl], f)
                    accu_id += 1
                pkl_files = []
            
            bar.update(1)

    if 0 !=  len(pkl_files):
        for pkl in pkl_files:
            pkl_name = '{}.pkl'.format(str(accu_id))
            with open(osp.join(save_dir, pkl_name), 'wb') as f:
                pickle.dump([pkl], f)
            accu_id += 1
        pkl_files = []
            
    return



def preprcess_nuScene(SEEM,SAM,dataset='',scene='',Resize=None):
    mapping = 'NuScenesLidarSegSCN'
    pkl_data = []

    curr_split = dataset
    # 1. Load Pickle Data
    print('load pkl data...')
    # data items, 
    # keys
    if 'USA_SING' == scene:
        save_path = save_new_usa_singapore_pkl_path
        if 'train' == curr_split:
            pkl_path_list = [pkl_path_train_usa_path,pkl_path_train_singapore_path] 
            if Resize:
                save_dir_list = [save_path +'/' + 'src' +'/' + curr_split + '_resize', save_path + '/' + 'trg' + '/' + curr_split + '_resize'] 
            else:
                save_dir_list = [save_path +'/' + 'src' +'/' + curr_split, save_path + '/' + 'trg' + '/' + curr_split] 
        elif 'test' == curr_split:
            pkl_path_list = [pkl_path_test_usa_path,pkl_path_test_singapore_path]
            save_dir_list = [save_path + '/' + 'src' + '/' + curr_split, save_path + '/' + 'trg' + '/' + curr_split]
        elif 'val' == curr_split:
            pkl_path_list = [pkl_path_val_singapore_path]
            save_dir_list = [save_path + '/' + curr_split]
        else:
            return
    elif 'DAY_NIGHT' == scene:
        save_path = save_new_day_night_pkl_path
        if 'train' == curr_split:
            pkl_path_list = [pkl_path_train_day_path,pkl_path_train_night_path]
            if Resize:
                save_dir_list = [save_path + '/' + 'src' + '/' + curr_split + '_resize', save_path + '/' + 'trg' + '/' + curr_split + '_resize']
            else:
                save_dir_list = [save_path + '/' + 'src' + '/' + curr_split, save_path + '/' + 'trg' + '/' + curr_split]
        elif 'test' == curr_split:
            pkl_path_list = [pkl_path_test_day_path,pkl_path_test_night_path]
            save_dir_list = [save_path + '/' + 'src' + '/' + curr_split, save_path + '/' + 'trg' + '/' + curr_split]
        elif 'val' == curr_split:
            pkl_path_list = [pkl_path_val_night_path]
            save_dir_list = [save_path + '/' + curr_split]
        else:
            return
    else:
        return
    

    for pkl_path,save_dir in zip(pkl_path_list,save_dir_list):
        print('process ' + scene + ' ' + curr_split + '...')
        os.makedirs(save_dir, exist_ok=True)
        with open(pkl_path,'rb') as f:
            pkl_data.extend(pickle.load(f))

        # 2. Iterate PKL
        print('iterate pkl data...')
        pkl_files = []
        accu_id = 0
        with tqdm(total=len(pkl_data)) as bar:
            # train data 18029 items, 
            for pkl_id, data in  enumerate(pkl_data):         
                new_pkl = {}
                img_path = osp.join(nuScene_orig_data_path, data['camera_path'])
                image = Image.open(img_path)

                if Resize:
                    image = image.resize(Resize, Image.BILINEAR)

        # 3. Generate Masks 
                sam_masks = SAM.generate(cv2.cvtColor(np.array(image).astype(np.uint8), cv2.COLOR_BGR2RGB))
                seem_masks,_ = call_SEEM(SEEM,pil_image=image,mapping=mapping)


                sam_masks = [x['segmentation'] for x in sam_masks]   # [mask, h, w]
                sam_masks = merge_sam_masks(sam_masks)  # Merge All mask into single 'Image'
                seem_masks =  [x for x in seem_masks.cpu().numpy()]

                new_pkl['sam'] = sam_masks.copy()
                new_pkl['seem'] = seem_masks.copy()

                pkl_files.append(new_pkl)

                if 0 == (len(pkl_files) % 10):
        # 4. Save PKL Data
        # save to pickle file
                    for pkl in pkl_files:
                        pkl_name = '{}.pkl'.format(str(accu_id))
                        with open(osp.join(save_dir, pkl_name), 'wb') as f:
                            pickle.dump([pkl], f)
                        accu_id += 1
                    pkl_files = []
                
                bar.update(1)

        if 0 !=  len(pkl_files):
            for pkl in pkl_files:
                pkl_name = '{}.pkl'.format(str(accu_id))
                with open(osp.join(save_dir, pkl_name), 'wb') as f:
                    pickle.dump([pkl], f)
                accu_id += 1
            pkl_files = []
            
    return



def test_data():
    test_pkl_dir = '/Labs/Scripts/3DPC/Datasets/3DPC/XMUDA_WITH_VFM/virtual_kitti_preprocess_vfm/train'
    num_a = np.random.randint(0,1000,1,dtype='int')
    num_b = np.random.randint(0,1000,1,dtype='int')
    pkl_file_path_a  = test_pkl_dir + '/' + str(num_a[0]) + '.pkl'
    pkl_file_path_b  = test_pkl_dir + '/' + str(num_b[0]) + '.pkl'

    pkl_data_a = []
    pkl_data_b = []

    # load data
    with open(pkl_file_path_a,'rb') as f:
        pkl_data_a.extend(pickle.load(f))

    with open(pkl_file_path_b,'rb') as f:
        pkl_data_b.extend(pickle.load(f))

    print('pkl a len = ',len(pkl_data_a))
    print('pkl b len = ',len(pkl_data_b))

    print('pkl a.sam len = ',len(pkl_data_a[0]['sam']))
    print('pkl b.sam len = ',len(pkl_data_b[0]['sam']))

    c = np.random.randint(0,len(pkl_data_a['seem']),1,dtype='int')
    print('pkl a.seem max = ',max(pkl_data_a[0]['seem'][c]))
    print('pkl b.seem max = ',max(pkl_data_b[0]['seem'][c]))

    return





if __name__ == "__main__":

    # test_data()

    print('load seem...')
    SEEM = build_SEEM(vfm_pth,vfm_cfg,cuda_device_idx) #cuda_device_idx
    print('load sam...')
    SAM  = build_SAM(device=cuda_device_idx) #cuda_device_idx
 

    if 'KITTI' in pre_defined_proc_list:
        preprocess_vkitti_train(SEEM,SAM)
        preprocess_skitti(SEEM,SAM,'train')
        preprocess_skitti(SEEM,SAM,'test')
        preprocess_skitti(SEEM,SAM,'val')

    if 'SKITTI_A2D2' in pre_defined_proc_list:
        preprocess_skitti(SEEM,SAM,'train',mapping='A2D2SCN')
        preprocess_skitti(SEEM,SAM,'test',mapping='A2D2SCN')
        preprocess_skitti(SEEM,SAM,'val',mapping='A2D2SCN')

    if 'A2D2' in  pre_defined_proc_list:
        preprcess_a2d2(SEEM,SAM,'train')
    
    if 'USA_SING' in pre_defined_proc_list:
        print('load USA_SING...')
        preprcess_nuScene(SEEM,SAM,'train','USA_SING')
        preprcess_nuScene(SEEM,SAM,'test','USA_SING')
        preprcess_nuScene(SEEM,SAM,'val','USA_SING')

    if 'DAY_NIGHT' in pre_defined_proc_list:
        print('load DAY_NIGHT...')
        preprcess_nuScene(SEEM,SAM,'train','DAY_NIGHT')
        preprcess_nuScene(SEEM,SAM,'test','DAY_NIGHT')
        preprcess_nuScene(SEEM,SAM,'val','DAY_NIGHT')


    if 'A2D2_Resize' in  pre_defined_proc_list:
        preprcess_a2d2(SEEM,SAM,'train',(480,302))

    if 'USA_SING_Resize' in pre_defined_proc_list:
        print('load USA_SING...')
        preprcess_nuScene(SEEM,SAM,'train','USA_SING',(400,225))

    if 'DAY_NIGHT_Resize' in pre_defined_proc_list:
        print('load DAY_NIGHT...')
        preprcess_nuScene(SEEM,SAM,'train','DAY_NIGHT',(400,225))