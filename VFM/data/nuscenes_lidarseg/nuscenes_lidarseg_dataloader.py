import os.path as osp
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

from xmuda.data.utils.refine_pseudo_labels import refine_pseudo_labels
from xmuda.data.utils.augmentation_3d import augment_and_scale_3d
from VFM.mix import sample_mix_masks, get_cut_masks

class NuScenesLidarSegBase(Dataset):
    """NuScenes dataset"""

    class_names = [
        'ignore',
        'barrier',
        'bicycle',
        'bus',
        'car',
        'construction_vehicle',
        'motorcycle',
        'pedestrian',
        'traffic_cone',
        'trailer',
        'truck',
        'driveable_surface',
        'other_flat',
        'sidewalk',
        'terrain',
        'manmade',
        'vegetation'
     ]

    # use those categories if merge_classes == True
    categories = {
        "vehicle": ["bicycle", "bus", "car", "construction_vehicle", "motorcycle", "trailer", "truck"],
        "driveable_surface": ["driveable_surface"],
        "sidewalk": ["sidewalk"],
        "terrain": ["terrain"],
        "manmade": ["manmade"],
        "vegetation": ["vegetation"],
        # "ignore": ["ignore", "barrier", "pedestrian", "traffic_cone", "other_flat"],
    }

    def __init__(self,
                 split,
                 preprocess_dir,
                 merge_classes=False,
                 pselab_paths=None
                 ):

        self.split = split
        self.preprocess_dir = preprocess_dir

        print("Initialize Nuscenes dataloader")

        assert isinstance(split, tuple)
        print('Load', split)
        self.data = []
        for curr_split in split:
            with open(osp.join(self.preprocess_dir, curr_split + '.pkl'), 'rb') as f:
                self.data.extend(pickle.load(f))

        self.pselab_data = None
        if pselab_paths:
            assert isinstance(pselab_paths, tuple)
            print('Load pseudo label data ', pselab_paths)
            self.pselab_data = []
            for curr_split in pselab_paths:
                self.pselab_data.extend(np.load(curr_split, allow_pickle=True))

            # check consistency of data and pseudo labels
            assert len(self.pselab_data) == len(self.data)
            for i in range(len(self.pselab_data)):
                assert len(self.pselab_data[i]['pseudo_label_2d']) == len(self.data[i]['seg_labels'])

            # refine 2d pseudo labels
            probs2d = np.concatenate([data['probs_2d'] for data in self.pselab_data])
            pseudo_label_2d = np.concatenate([data['pseudo_label_2d'] for data in self.pselab_data]).astype(np.int)
            pseudo_label_2d = refine_pseudo_labels(probs2d, pseudo_label_2d)

            # refine 3d pseudo labels
            # fusion model has only one final prediction saved in probs_2d
            if self.pselab_data[0]['probs_3d'] is not None:
                probs3d = np.concatenate([data['probs_3d'] for data in self.pselab_data])
                pseudo_label_3d = np.concatenate([data['pseudo_label_3d'] for data in self.pselab_data]).astype(np.int)
                pseudo_label_3d = refine_pseudo_labels(probs3d, pseudo_label_3d)
            else:
                pseudo_label_3d = None

            # undo concat
            left_idx = 0
            for data_idx in range(len(self.pselab_data)):
                right_idx = left_idx + len(self.pselab_data[data_idx]['probs_2d'])
                self.pselab_data[data_idx]['pseudo_label_2d'] = pseudo_label_2d[left_idx:right_idx]
                if pseudo_label_3d is not None:
                    self.pselab_data[data_idx]['pseudo_label_3d'] = pseudo_label_3d[left_idx:right_idx]
                else:
                    self.pselab_data[data_idx]['pseudo_label_3d'] = None
                left_idx = right_idx

        if merge_classes:
            self.label_mapping = -100 * np.ones(len(self.class_names), dtype=int)
            for cat_idx, cat_list in enumerate(self.categories.values()):
                for class_name in cat_list:
                    self.label_mapping[self.class_names.index(class_name)] = cat_idx
            self.class_names = list(self.categories.keys())
        else:
            self.label_mapping = None

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)


class NuScenesLidarSegSCN(NuScenesLidarSegBase):
    def __init__(self,
                 args,
                 split,
                 preprocess_dir,
                 nuscenes_dir='',
                 pselab_paths=None,
                 vfm_data_paths='',
                 merge_classes=False,
                 scale=20,
                 full_scale=4096,
                 resize=(400, 225),
                 image_normalizer=None,
                 noisy_rot=0.0,  # 3D augmentation
                 flip_x=0.0,  # 3D augmentation
                 rot_z=0.0,  # 3D augmentation
                 transl=False,  # 3D augmentation
                 fliplr=0.0,  # 2D augmentation
                 color_jitter=None,  # 2D augmentation
                 output_orig=False
                 ):
        super().__init__(split,
                         preprocess_dir,
                         merge_classes=merge_classes,
                         pselab_paths=pselab_paths)

        self.nuscenes_dir = nuscenes_dir
        self.output_orig = output_orig

        # point cloud parameters
        self.scale = scale
        self.full_scale = full_scale
        # 3D augmentation
        self.noisy_rot = noisy_rot
        self.flip_x = flip_x
        self.rot_z = rot_z
        self.transl = transl

        # image parameters
        self.resize = resize
        self.image_normalizer = image_normalizer

        # data augmentation
        self.fliplr = fliplr
        self.color_jitter = T.ColorJitter(*color_jitter) if color_jitter else None

        self.with_vfm = args.vfmlab
        self.vfm_data_paths = vfm_data_paths

    def __getitem__(self, index):
        data_dict = self.data[index]

        points = data_dict['points'].copy()
        seg_label = data_dict['seg_labels'].astype(np.int64)

        if self.label_mapping is not None:
            seg_label = self.label_mapping[seg_label]

        # load pre-computed vfm segmentation data
        vfm_dict = []
        with open(osp.join(self.vfm_data_paths, str(index) + '.pkl'), 'rb') as f:
            vfm_dict.extend(pickle.load(f))

        out_dict = {}

        points_img = data_dict['points_img'].copy()
        img_path = osp.join(self.nuscenes_dir, data_dict['camera_path'])
        image = Image.open(img_path)
        # image_orig = image.copy()

        # load pre-computed VFM mask
        masks_sam  = vfm_dict[-1]['sam']  # Merged Masks
        masks_seem = vfm_dict[-1]['seem'] # Class-wise Masks (class, h, w)

        # Get Original points image
        # if self.with_vfm:
        #     points_img_orig = data_dict['points_img'].copy()
        #     points_img_orig[:,0] = np.floor(points_img_orig[:, 0])
        #     points_img_orig[:,1] = np.floor(points_img_orig[:, 1])
        #     img_indices_org = points_img_orig.astype(np.int64)
        #     assert np.all(img_indices_org[:, 0] >= 0)
        #     assert np.all(img_indices_org[:, 1] >= 0)
        #     assert np.all(img_indices_org[:, 0] < image.size[1]) # image_orig
        #     assert np.all(img_indices_org[:, 1] < image.size[0]) # image_orig
        #     image_orig = image.copy()

        if self.resize:
            if not image.size == self.resize:
                # check if we do not enlarge downsized images
                assert image.size[0] > self.resize[0]

                # scale image points
                points_img[:, 0] = float(self.resize[1]) / image.size[1] * np.floor(points_img[:, 0]) # Height 900
                points_img[:, 1] = float(self.resize[0]) / image.size[0] * np.floor(points_img[:, 1]) # Width  1600

                # resize image
                image = image.resize(self.resize, Image.BILINEAR)

        img_indices = points_img.astype(np.int64)

        assert np.all(img_indices[:, 0] >= 0)
        assert np.all(img_indices[:, 1] >= 0)
        assert np.all(img_indices[:, 0] < image.size[1])
        assert np.all(img_indices[:, 1] < image.size[0])

        # 2D augmentation
        if self.color_jitter is not None:
            image = self.color_jitter(image)
        # PIL to numpy
        image = np.array(image, dtype=np.float32, copy=False) / 255.
        # image_orig = np.array(image_orig, dtype=np.float32, copy=False) / 255.
        # 2D augmentation
        if np.random.rand() < self.fliplr:
            image = np.ascontiguousarray(np.fliplr(image))
            img_indices[:, 1] = image.shape[1] - 1 - img_indices[:, 1]

        # normalize image
        if self.image_normalizer:
            mean, std = self.image_normalizer
            mean = np.asarray(mean, dtype=np.float32)
            std = np.asarray(std, dtype=np.float32)
            image = (image - mean) / std

        out_dict['img'] = np.moveaxis(image, -1, 0)
        out_dict['img'] = torch.from_numpy(out_dict['img'])
        out_dict['img_indices'] = img_indices

       # 3D data augmentation and scaling from points to voxel indices
        # nuscenes lidar coordinates: x (right), y (front), z (up)
        coords = augment_and_scale_3d(points, self.scale, self.full_scale, noisy_rot=self.noisy_rot,
                                      flip_x=self.flip_x, rot_z=self.rot_z, transl=self.transl)

        # cast to integer
        coords = coords.astype(np.int64)

        # only use voxels inside receptive field
        idxs = (coords.min(1) >= 0) * (coords.max(1) < self.full_scale)

        out_dict['coords'] = torch.from_numpy(coords[idxs])
        out_dict['feats'] = torch.from_numpy(np.ones([len(idxs), 1], np.float32))  # simply use 1 as feature
        out_dict['seg_label'] = torch.from_numpy(seg_label[idxs])

        out_dict['img_indices'] = out_dict['img_indices'][idxs]

        sampled_sam_mask,sampled_sam_indices = sample_mix_masks(masks_sam, out_dict['img_indices']) #,out_dict['seg_label'], out_dict['coords'] 
        cut_mix_mask,cut_mix_indices = get_cut_masks(out_dict['img'], out_dict['img_indices'])

        if self.pselab_data is not None:
            out_dict['pseudo_label_2d'] = torch.from_numpy(self.pselab_data[index]['pseudo_label_2d'][idxs])
            out_dict['sam_mix_pseudo_label_2d'] = out_dict['pseudo_label_2d'].clone()
            out_dict['cut_mix_pseudo_label_2d'] = out_dict['pseudo_label_2d'].clone()
            out_dict['sam_pseudo_label_2d'] = (out_dict['pseudo_label_2d'].clone())[sampled_sam_indices] # Slice SAM Sampled label (Num, label)
            out_dict['cut_pseudo_label_2d'] = (out_dict['pseudo_label_2d'].clone())[cut_mix_indices] # Slice Cut Sampled label (Num, label)

            out_dict['sam_mix_indices'] = out_dict['img_indices']
            out_dict['cut_mix_indices'] = out_dict['img_indices']
            
            if self.pselab_data[index]['pseudo_label_3d'] is None:
                out_dict['pseudo_label_3d'] = None
                out_dict['sam_mix_pseudo_label_3d'] = None
                out_dict['cut_mix_pseudo_label_3d'] = None
            else:
                out_dict['pseudo_label_3d'] = torch.from_numpy(self.pselab_data[index]['pseudo_label_3d'][idxs])
                out_dict['sam_mix_pseudo_label_3d'] = out_dict['pseudo_label_3d'].clone()
                out_dict['cut_mix_pseudo_label_3d'] = out_dict['pseudo_label_3d'].clone()            
                out_dict['sam_pseudo_label_3d'] = (out_dict['pseudo_label_3d'].clone())[sampled_sam_indices] # Slice SAM Sampled label (Num, label)
                out_dict['cut_pseudo_label_3d'] = (out_dict['pseudo_label_3d'].clone())[cut_mix_indices] # Slice Cut Sampled label (Num, label)

        if self.output_orig:
            out_dict.update({
                'orig_seg_label': seg_label,
                'orig_points_idx': idxs,
            })

        '''
        Prepare Mixed Data
        '''
        out_dict['masks_seem']= torch.from_numpy(np.stack(masks_seem))

        # SAM Mix
        # Mask
        out_dict['sampled_sam_mask'] = torch.from_numpy(sampled_sam_mask)
        out_dict['sampled_sam_sel_indices'] = sampled_sam_indices   # select indices

        # 2D Data
        out_dict['sam_mix_image'] = out_dict['img'].clone().permute(1,2,0) # (H,W,C)
        out_dict['sam_mix_label_2d'] = out_dict['seg_label'].clone()
        out_dict['sam_mix_indices'] = out_dict['img_indices']
        out_dict['sam_label'] =  (out_dict['seg_label'].clone())[sampled_sam_indices] # Slice SAM Sampled label (Num, label)
        
        # 3D Data
        out_dict['sam_mix_coords'] = out_dict['coords'].clone()
        out_dict['sam_mix_feats'] = out_dict['feats'].clone()
        out_dict['sam_mix_label_3d'] = out_dict['seg_label'].clone()
        
        # CutMix
        # Mask
        out_dict['cut_mask'] = torch.from_numpy(cut_mix_mask)
        out_dict['cut_sel_indices'] = cut_mix_indices   # select indices

        # 2D Data
        out_dict['cut_mix_image'] = out_dict['img'].clone().permute(1,2,0) # (H,W,C)
        out_dict['cut_mix_label_2d'] = out_dict['seg_label'].clone()
        out_dict['cut_mix_indices'] = out_dict['img_indices']
        out_dict['cut_label'] = (out_dict['seg_label'].clone())[cut_mix_indices] # Slice Cut Sampled label (Num, label)
        
        # 3D Data
        out_dict['cut_mix_coords'] = out_dict['coords'].clone()
        out_dict['cut_mix_feats'] = out_dict['feats'].clone()
        out_dict['cut_mix_label_3d'] = out_dict['seg_label'].clone() 

        out_dict['img'] = out_dict['img'].permute(1,2,0)

        return out_dict


def test_NuScenesSCN():
    from xmuda.data.utils.visualize import draw_points_image_labels, draw_points_image_depth, draw_bird_eye_view
    preprocess_dir = '/datasets_local/datasets_mjaritz/nuscenes_lidarseg_preprocess/preprocess'
    nuscenes_dir = '/datasets_local/datasets_mjaritz/nuscenes_preprocess'
    # # split = ('train_singapore',)
    # # pselab_paths = ('/home/docker_user/workspace/outputs/xmuda/nuscenes/usa_singapore/xmuda/pselab_data/train_singapore.npy',)
    # # split = ('train_night',)
    # # pselab_paths = ('/home/docker_user/workspace/outputs/xmuda/nuscenes/day_night/xmuda/pselab_data/train_night.npy',)
    # split = ('val_night',)
    split = ('test_singapore',)
    dataset = NuScenesLidarSegSCN(split=split,
                                  preprocess_dir=preprocess_dir,
                                  nuscenes_dir=nuscenes_dir,
                                  # pselab_paths=pselab_paths,
                                  merge_classes=True,
                                  noisy_rot=0.1,
                                  flip_x=0.5,
                                  rot_z=2*np.pi,
                                  transl=True,
                                  fliplr=0.5,
                                  color_jitter=(0.4, 0.4, 0.4)
                                  )
    for i in [10, 20, 30, 40, 50, 60]:
        data = dataset[i]
        coords = data['coords']
        seg_label = data['seg_label']
        img = np.moveaxis(data['img'], 0, 2)
        img_indices = data['img_indices']
        draw_points_image_labels(img, img_indices, seg_label, color_palette_type='NuScenesLidarSeg', point_size=3)
        # pseudo_label_2d = data['pseudo_label_2d']
        # draw_points_image_labels(img, img_indices, pseudo_label_2d, color_palette_type='NuScenes', point_size=3)
        # draw_bird_eye_view(coords)
        print('Number of points:', len(coords))


def compute_class_weights():
    preprocess_dir = '/datasets_local/datasets_mjaritz/nuscenes_lidarseg_preprocess/preprocess'
    split = ('train_usa', 'test_usa')  # nuScenes-lidarseg USA/Singapore
    # split = ('train_day', 'test_day')  # nuScenes-lidarseg Day/Night
    # split = ('train_singapore', 'test_singapore')  # nuScenes-lidarseg Singapore/USA
    # split = ('train_night', 'test_night')  # nuScenes-lidarseg Night/Day
    # split = ('train_singapore_labeled',)  # SSDA: nuScenes-lidarseg Singapore labeled
    dataset = NuScenesLidarSegBase(split,
                                   preprocess_dir,
                                   merge_classes=True
                                   )
    # compute points per class over whole dataset
    num_classes = len(dataset.class_names)
    points_per_class = np.zeros(num_classes, int)
    for i, data in enumerate(dataset.data):
        print('{}/{}'.format(i, len(dataset)))
        labels = dataset.label_mapping[data['seg_labels']]
        points_per_class += np.bincount(labels[labels != -100], minlength=num_classes)

    # compute log smoothed class weights
    class_weights = np.log(5 * points_per_class.sum() / points_per_class)
    print('log smoothed class weights: ', class_weights / class_weights.min())


def compute_stats():
    preprocess_dir = 'path/to/data/nuscenes_lidarseg_preprocess/preprocess'
    nuscenes_dir = 'path/to/data/nuscenes'
    outdir = 'path/to/data/nuscenes_lidarseg_preprocess/stats'
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    splits = ('train_day', 'test_day', 'train_night', 'val_night', 'test_night',
              'train_usa', 'test_usa', 'train_singapore', 'val_singapore', 'test_singapore')
    for split in splits:
        dataset = NuScenesLidarSegSCN(
            split=(split,),
            preprocess_dir=preprocess_dir,
            nuscenes_dir=nuscenes_dir
        )
        # compute points per class over whole dataset
        num_classes = len(dataset.class_names)
        points_per_class = np.zeros(num_classes, int)
        for i, data in enumerate(dataset.data):
            print('{}/{}'.format(i, len(dataset)))
            points_per_class += np.bincount(data['seg_labels'], minlength=num_classes)

        plt.barh(dataset.class_names, points_per_class)
        plt.grid(axis='x')

        # add values right to the bar plot
        for i, value in enumerate(points_per_class):
            x_pos = value
            y_pos = i
            if dataset.class_names[i] == 'driveable_surface':
                x_pos -= 0.25 * points_per_class.max()
                y_pos += 0.75
            plt.text(x_pos + 0.02 * points_per_class.max(), y_pos - 0.25, f'{value:,}', color='blue', fontweight='bold')
        plt.title(split)
        plt.tight_layout()
        # plt.show()
        plt.savefig(outdir / f'{split}.png')
        plt.close()


if __name__ == '__main__':
    # test_NuScenesSCN()
    compute_class_weights()
    # compute_stats()
