import os
import warnings
import PIL
from PIL import Image
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

# import gradio as gr
import torch
import argparse
# import whisper
import numpy as np

# from gradio import processing_utils
from xdecoder.BaseModel import BaseModel
from xdecoder import build_model
from utils.distributed import init_distributed
from utils.arguments import load_opt_from_config_files
from utils.constants import COCO_PANOPTIC_CLASSES

import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from utils.visualizer import Visualizer


from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from detectron2.data import MetadataCatalog

from class_mapping import COCO_TO_NuScenes, COCO_TO_A2D2_SKITTI, COCO_TO_VKITTI_SKITTI

import csv


def map_classes(src_logits,src_label,des_labels):
    return



@torch.no_grad()
def inference(model, image, tasks, info, audio=None, refimg=None, reftxt=None, audio_pth=None, video_pth=None):
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        return interface_seem(model, audio, image, tasks, info, refimg, reftxt, audio_pth, video_pth)



def interface_seem(model, audio, image, tasks, info, refimg=None, reftxt=None, audio_pth=None, video_pth=None):

    t = []
    t.append(transforms.Resize(512, interpolation=Image.BICUBIC))
    transform = transforms.Compose(t)
    metadata = MetadataCatalog.get('coco_2017_train_panoptic')
    all_classes = [name.replace('-other','').replace('-merged','') for name in COCO_PANOPTIC_CLASSES] + ["others"]
    colors_list = [(np.array(color['color'])/255).tolist() for color in COCO_CATEGORIES] + [[1, 1, 1]]

    image_ori = image

    mapping = info[0]
    prompt  = info[1]

    width = image_ori.size[0]
    height = image_ori.size[1]
    image_ori = np.asarray(image_ori)
    visual = Visualizer(image_ori, metadata=metadata)
    images = torch.from_numpy(image_ori.copy()).permute(2,0,1).cuda()


    data = {"image": images, "height": height, "width": width}
    if len(tasks) == 0:
        tasks = ["Panoptic"]
    
    # inistalize task
    model.model.task_switch['spatial'] = False
    model.model.task_switch['visual'] = False
    model.model.task_switch['grounding'] = False
    model.model.task_switch['audio'] = False

    example = None
    stroke = None      
    text = None    
    audio = None
    
    batch_inputs = [data]
    if 'Panoptic' in tasks:
        model.model.metadata = metadata
        results = model.model.evaluate(batch_inputs)
        pano_seg = results[-1]['panoptic_seg'][0]       #   0-14 Mask ID (900,1600) for each pixel
        pano_seg_info = results[-1]['panoptic_seg'][1]  #   1-14 Mask ID, Category ID
        pano_seg_logits = results[-1]['panoptic_seg'][2].permute(2,0,1)

        
        if mapping == 'NuScenesLidarSegSCN':
            pano_seg_logits = pano_seg_logits[COCO_TO_NuScenes['masks']]
            mapped_labels = np.array(COCO_TO_NuScenes['classes'])[COCO_TO_NuScenes['masks']]
            class_num = COCO_TO_NuScenes['class_num']
            CoCoMap = COCO_TO_NuScenes['Mapping']
        elif mapping == 'A2D2SCN':
            pano_seg_logits = pano_seg_logits[COCO_TO_A2D2_SKITTI['masks']]
            mapped_labels = np.array(COCO_TO_A2D2_SKITTI['classes'])[COCO_TO_A2D2_SKITTI['masks']]
            class_num = COCO_TO_A2D2_SKITTI['class_num']
            CoCoMap = COCO_TO_A2D2_SKITTI['Mapping']
        elif mapping == 'SemanticKITTISCN':
            pano_seg_logits = pano_seg_logits[COCO_TO_VKITTI_SKITTI['masks']]
            mapped_labels = np.array(COCO_TO_VKITTI_SKITTI['classes'])[COCO_TO_VKITTI_SKITTI['masks']]
            class_num = COCO_TO_VKITTI_SKITTI['class_num']
            CoCoMap = COCO_TO_VKITTI_SKITTI['Mapping']
        else:
            pass
        # Merge duplicate masks
        res = torch.zeros((class_num, pano_seg_logits.shape[1], pano_seg_logits.shape[2]), 
                          dtype=torch.half, device=pano_seg_logits.device)  -100.

        for i in range(class_num):
            mask = mapped_labels == i
            check = set(mask)
            if 1 == len(check) and False in check:
                continue
            res[i] = torch.mean(pano_seg_logits[mask],axis=0) # merge logits by averaging similar classes

        
        # Temp Mapping : utilize pano_seg
        for mask in pano_seg_info:
            k = mask['category_id'] + 1
            if  k in CoCoMap.keys():
                pano_seg[pano_seg==mask['id']] = CoCoMap[k]
            else:
                pano_seg[pano_seg==mask['id']] = -100
            
        return res,pano_seg #pano_seg_logits
        

    else:
        results,image_size,extra = model.model.evaluate_demo(batch_inputs)


    res = []

    return Image.fromarray(res), None


def call_SEEM(seem_model,image_path=None,np_image=None,pil_image=None,mapping = None, prompt = None):
    if prompt is not None:
        task = ['stroke']
    else: 
        task = []
    if np_image:
        image = Image.fromarray(np_image)
        return inference(seem_model,image,task,(mapping,prompt))

    if image_path:
        image = Image.open(image_path)
        return inference(seem_model,image,task,(mapping,prompt))

    if pil_image:
        return inference(seem_model,pil_image,task,(mapping,prompt))

    return None

def build_SEEM(pretrained_pth,config_pth,gpu_index):
    '''
    build args
    '''
    opt = load_opt_from_config_files(config_pth)
    opt = init_distributed(opt,gpu_index)

    '''
    build model
    '''
    model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()
    with torch.no_grad():
        model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(COCO_PANOPTIC_CLASSES + ["background"], is_eval=True)
    
    return model


def parse_option():
    parser = argparse.ArgumentParser('SEEM Interface', add_help=False)
    print(os.getcwd())
    parser.add_argument('--conf_files', default="../VFM/configs/seem/seem_focall_lang.yaml", metavar="FILE", help='path to config file', )
    args = parser.parse_args()

    return args


def main():
    '''
    build args
    '''
    args = parse_option()
    opt = load_opt_from_config_files(args.conf_files)
    opt = init_distributed(opt)

    # META DATA
    cur_model = 'None'
    # if 'focalt' in args.conf_files:
    #     pretrained_pth = os.path.join("seem_focalt_v2.pt")
    #     if not os.path.exists(pretrained_pth):
    #         os.system("wget {}".format("https://huggingface.co/xdecoder/SEEM/resolve/main/seem_focalt_v2.pt"))
    #     cur_model = 'Focal-T'
    # elif 'focal' in args.conf_files:
    #     pretrained_pth = os.path.join("seem_focall_v1.pt")
    #     if not os.path.exists(pretrained_pth):
    #         os.system("wget {}".format("https://huggingface.co/xdecoder/SEEM/resolve/main/seem_focall_v1.pt"))
    #     cur_model = 'Focal-L'
    pretrained_pth = "/Labs/Scripts/3DPC/xMUDA/VFM_ckpt/seem_focall_v1.pt"
    cur_model = 'Focal-L'

    '''
    build model
    '''
    model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()
    with torch.no_grad():
        model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(COCO_PANOPTIC_CLASSES + ["background"], is_eval=True)

    '''
    audio
    '''
    # audio = whisper.load_model("base")
    audio = None

    # Test Interface
    task = []
    # image = Image.open('/Labs/Scripts/3DPC/exp_xmuda_journal/out/nuscenes_lidarseg/usa_singapore/uda/xmuda/images/n015-2018-08-02-17-16-37+0800__CAM_FRONT__1533201470412460.jpg')
    image = Image.open('/Labs/Scripts/3DPC/Datasets/3DPC/nuScenes/samples/CAM_FRONT/n015-2018-08-02-17-16-37+0800__CAM_FRONT__1533201470412460.jpg')
    res = inference(model,audio,image,task,)
    
    # res.save('/Labs/Scripts/3DPC/exp_xmuda_journal/out/nuscenes_lidarseg/usa_singapore/uda/xmuda/images/n015-2018-08-02-17-16-37+0800__CAM_FRONT__1533201470412460-SEEM.jpg')
    # res.save('/Labs/Scripts/3DPC/exp_xmuda_journal/out/nuscenes_lidarseg/usa_singapore/uda/xmuda/images/n015-2018-08-02-17-16-37+0800__CAM_FRONT__1533201470412460-SEEM-ORG.jpg')

if __name__ == '__main__':
    main()