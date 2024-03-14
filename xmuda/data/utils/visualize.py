import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict

from xmuda.data.utils.turbo_cmap import interpolate_or_clip, turbo_colormap_data


# all classes
NUSCENES_COLOR_PALETTE = [
    (255, 158, 0),  # car
    (255, 158, 0),  # truck
    (255, 158, 0),  # bus
    (255, 158, 0),  # trailer
    (255, 158, 0),  # construction_vehicle
    (0, 0, 230),  # pedestrian
    (255, 61, 99),  # motorcycle
    (255, 61, 99),  # bicycle
    (0, 0, 0),  # traffic_cone
    (0, 0, 0),  # barrier
    (200, 200, 200),  # background
]

# classes after merging (as used in xMUDA)
NUSCENES_COLOR_PALETTE_SHORT = [
    (255, 158, 0),  # vehicle
    (0, 0, 230),  # pedestrian
    (255, 61, 99),  # bike
    (0, 0, 0),  # traffic boundary
    (200, 200, 200),  # background
]

NUSCENES_LIDARSEG_COLOR_PALETTE_DICT = OrderedDict([
    ('ignore', (0, 0, 0)),  # Black
    ('barrier', (112, 128, 144)),  # Slategrey
    ('bicycle', (220, 20, 60)),  # Crimson
    ('bus', (255, 127, 80)),  # Coral
    ('car', (255, 158, 0)),  # Orange
    ('construction_vehicle', (233, 150, 70)),  # Darksalmon
    ('motorcycle', (255, 61, 99)),  # Red
    ('pedestrian', (0, 0, 230)),  # Blue
    ('traffic_cone', (47, 79, 79)),  # Darkslategrey
    ('trailer', (255, 140, 0)),  # Darkorange
    ('truck', (255, 99, 71)),  # Tomato
    ('driveable_surface', (0, 207, 191)),  # nuTonomy green
    ('other_flat', (175, 0, 75)),
    ('sidewalk', (75, 0, 75)),
    ('terrain', (112, 180, 60)),
    ('manmade', (222, 184, 135)),  # Burlywood
    ('vegetation', (0, 175, 0))  # Green
])

NUSCENES_LIDARSEG_COLOR_PALETTE = list(NUSCENES_LIDARSEG_COLOR_PALETTE_DICT.values())

NUSCENES_LIDARSEG_COLOR_PALETTE_SHORT = [
    NUSCENES_LIDARSEG_COLOR_PALETTE_DICT['car'],  # vehicle
    NUSCENES_LIDARSEG_COLOR_PALETTE_DICT['driveable_surface'],
    NUSCENES_LIDARSEG_COLOR_PALETTE_DICT['sidewalk'],
    NUSCENES_LIDARSEG_COLOR_PALETTE_DICT['terrain'],
    NUSCENES_LIDARSEG_COLOR_PALETTE_DICT['manmade'],
    NUSCENES_LIDARSEG_COLOR_PALETTE_DICT['vegetation'],
    NUSCENES_LIDARSEG_COLOR_PALETTE_DICT['ignore']
]


# all classes
A2D2_COLOR_PALETTE_SHORT = [
    (255, 0, 0),  # car
    (255, 128, 0),  # truck
    (182, 89, 6),  # bike
    (204, 153, 255),  # person
    (255, 0, 255),  # road
    (150, 150, 200),  # parking
    (180, 150, 200),  # sidewalk
    (241, 230, 255),  # building
    (147, 253, 194),  # nature
    (255, 246, 143),  # other-objects
    (0, 0, 0)  # ignore
]

# colors as defined in https://github.com/PRBonn/semantic-kitti-api/blob/master/config/semantic-kitti.yaml
SEMANTIC_KITTI_ID_TO_BGR = {  # bgr
  0: [0, 0, 0],
  1: [0, 0, 255],
  10: [245, 150, 100],
  11: [245, 230, 100],
  13: [250, 80, 100],
  15: [150, 60, 30],
  16: [255, 0, 0],
  18: [180, 30, 80],
  20: [255, 0, 0],
  30: [30, 30, 255],
  31: [200, 40, 255],
  32: [90, 30, 150],
  40: [255, 0, 255],
  44: [255, 150, 255],
  48: [75, 0, 75],
  49: [75, 0, 175],
  50: [0, 200, 255],
  51: [50, 120, 255],
  52: [0, 150, 255],
  60: [170, 255, 150],
  70: [0, 175, 0],
  71: [0, 60, 135],
  72: [80, 240, 150],
  80: [150, 240, 255],
  81: [0, 0, 255],
  99: [255, 255, 50],
  252: [245, 150, 100],
  256: [255, 0, 0],
  253: [200, 40, 255],
  254: [30, 30, 255],
  255: [90, 30, 150],
  257: [250, 80, 100],
  258: [180, 30, 80],
  259: [255, 0, 0],
}
SEMANTIC_KITTI_COLOR_PALETTE = [SEMANTIC_KITTI_ID_TO_BGR[id] if id in SEMANTIC_KITTI_ID_TO_BGR.keys() else [0, 0, 0]
                                for id in range(list(SEMANTIC_KITTI_ID_TO_BGR.keys())[-1] + 1)]


# classes after merging (as used in xMUDA)
SEMANTIC_KITTI_COLOR_PALETTE_SHORT_BGR = [
    [245, 150, 100],  # car
    [180, 30, 80],  # truck
    [150, 60, 30],  # bike
    [30, 30, 255],  # person
    [255, 0, 255],  # road
    [255, 150, 255],  # parking
    [75, 0, 75],  # sidewalk
    [0, 200, 255],  # building
    [0, 175, 0],  # nature
    [255, 255, 50],  # other-objects
    [0, 0, 0],  # ignore
]
SEMANTIC_KITTI_COLOR_PALETTE_SHORT = [(c[2], c[1], c[0]) for c in SEMANTIC_KITTI_COLOR_PALETTE_SHORT_BGR]

VIRTUAL_KITTI_COLOR_PALETTE = [
    [0, 175, 0],  # vegetation_terrain
    [255, 200, 0],  # building
    [255, 0, 255],  # road
    [50, 255, 255],  # other-objects
    [80, 30, 180],  # truck
    [100, 150, 245],  # car
    [0, 0, 0],  # ignore
]

WAYMO_COLOR_PALETTE = [
    (200, 200, 200),  # unknown
    (255, 158, 0),  # vehicle
    (0, 0, 230),  # pedestrian
    (50, 255, 255),  # sign
    (255, 61, 99),  # cyclist
    (0, 0, 0),  # ignore
]


def draw_points_image_labels(img, img_indices, seg_labels, show=True, color_palette_type='NuScenes', point_size=0.5):
    if color_palette_type == 'NuScenes':
        color_palette = NUSCENES_COLOR_PALETTE_SHORT
    elif color_palette_type == 'NuScenesLidarSeg':
        color_palette = NUSCENES_LIDARSEG_COLOR_PALETTE_SHORT
    elif color_palette_type == 'NuScenesLidarSegLong':
        color_palette = NUSCENES_LIDARSEG_COLOR_PALETTE
    elif color_palette_type == 'A2D2':
        color_palette = A2D2_COLOR_PALETTE_SHORT
    elif color_palette_type == 'SemanticKITTI':
        color_palette = SEMANTIC_KITTI_COLOR_PALETTE_SHORT
    elif color_palette_type == 'SemanticKITTI_long':
        color_palette = SEMANTIC_KITTI_COLOR_PALETTE
    elif color_palette_type == 'VirtualKITTI':
        color_palette = VIRTUAL_KITTI_COLOR_PALETTE
    elif color_palette_type == 'Waymo':
        color_palette = WAYMO_COLOR_PALETTE
    else:
        raise NotImplementedError('Color palette type not supported')
    color_palette = np.array(color_palette) / 255.
    seg_labels[seg_labels == -100] = len(color_palette) - 1
    colors = color_palette[seg_labels]

    plt.imshow(img)
    plt.scatter(img_indices[:, 1], img_indices[:, 0], c=colors, alpha=0.5, s=point_size)

    plt.axis('off')
    plt.tight_layout()

    if show:
        plt.show()


def normalize_depth(depth, d_min, d_max):
    # normalize linearly between d_min and d_max
    data = np.clip(depth, d_min, d_max)
    return (data - d_min) / (d_max - d_min)


def draw_points_image_depth(img, img_indices, depth, show=True, point_size=0.5):
    # depth = normalize_depth(depth, d_min=3., d_max=50.)
    depth = normalize_depth(depth, d_min=depth.min(), d_max=depth.max())
    colors = []
    for depth_val in depth:
        colors.append(interpolate_or_clip(colormap=turbo_colormap_data, x=depth_val))
    # ax5.imshow(np.full_like(img, 255))
    plt.imshow(img)
    plt.scatter(img_indices[:, 1], img_indices[:, 0], c=colors, alpha=0.5, s=point_size)

    plt.axis('off')

    if show:
        plt.show()


def draw_bird_eye_view(coords, full_scale=4096):
    plt.scatter(coords[:, 0], coords[:, 1], s=0.1)
    plt.xlim([0, full_scale])
    plt.ylim([0, full_scale])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


def save_2D_segmentations(org_image,anns,path=None,points=None,img_indices=None,batch_idx = 0,org_labels=None,class_num = 6,transparency=0.8,point_size = 0.8):
    if len(anns) == 0:
        return
    org_image = org_image.permute(1,2,0)
    plt.figure(figsize=(20,35))
    plt.imshow(org_image)


    if org_labels is not None:
        colors = []
        color_points = [ [1,0.62,0.001,0.9],        # 0  (0.8)
                    [0.001,0.81,0.6,0.9],          # 1
                    [0.29,0.001,0.29,0.9],         # 2
                    [0.44,0.71,0.24,0.9],          # 3
                    [0.95,0.001,0.001,0.9],          # 4    0.87,0.72,0.53,0.9
                    [0.001,0.69,0.001,0.9],        # 5
                    [0.0,0.0,0.0,0.35]]
        
        for label in org_labels:
            if label == -100:
                colors.append(color_points[-1])
            else:
                colors.append(color_points[label])

        plt.scatter(img_indices[batch_idx][:, 1], img_indices[batch_idx][:, 0], c=colors, alpha=1, s=1)

    elif points is not None and img_indices is not None:
        depth = points[:,2]
        depth = normalize_depth(depth, d_min=depth.min(), d_max=depth.max())
        colors = []
        for depth_val in depth:
            colors.append(interpolate_or_clip(colormap=turbo_colormap_data, x=depth_val))
        plt.scatter(img_indices[batch_idx][:, 1], img_indices[batch_idx][:, 0], c=colors, alpha=transparency, s=point_size)
    else:
        pass

    ax = plt.gca()
    # ax.set_autoscale_on(False)

    img_mask = np.ones((anns.shape[0], anns.shape[1], 4))
    img_mask[:,:,3] = 0

    # 225,400
    # Different Class
    color_masks = [ [1,0.62,0.001,0.2],             # 0  (0.8)
                    [0.001,0.81,0.6,0.2],          # 1
                    [0.29,0.001,0.29,0.2],         # 2
                    [0.44,0.71,0.24,0.2],          # 3
                    [0.95,0.001,0.001,0.2],          # 4    [0.87,0.72,0.53,0.2]
                    [0.001,0.69,0.001,0.2],        # 5
                    [0.1,0.9,0.3,0.35], 
                    [0.2,0.7,0.4,0.35],
                    [0.3,0.5,0.5,0.35],
                    [0.4,0.4,0.6,0.35],
                    [0.5,0.3,0.7,0.35],
                    [0.6,0.2,0.8,0.35],
                    [0.7,0.8,0.6,0.35],
                    [0.8,0.3,0.4,0.35],
                    [0.3,0.3,0.9,0.35],
                    [0.0,0.0,0.0,0.35]]
    for i in range(0,class_num):
        mask = anns == i
        # color_mask = np.concatenate([np.random.random(3), [0.35]])
        # img_mask[mask] = color_mask

        img_mask[mask] = color_masks[i] # (when class == -1, paint the last color)

    # mask = anns == 5
    # img_mask[mask] = color_masks[5]
    
    ax.imshow(img_mask)
    plt.axis('off')
    
    if path is not None:
        plt.savefig(path)
        # plt.savefig('/Labs/Scripts/3DPC/exp_xmuda_journal/out/nuscenes_lidarseg/usa_singapore/uda/xmuda/images/seg_res-n015-2018-08-02-17-16-37+0800__CAM_FRONT__1533201470412460.jpg')

