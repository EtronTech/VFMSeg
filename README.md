# Visual Foundation Models Boost Cross-Modal Unsupervised Domain Adaptation for 3D Semantic Segmentation
# Code Samples

## VFMs
We use released official code of [SAM](https://segment-anything.com/) and [SEEM](https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once) including their checkpoints (`sam_vit_h_4b8939.pth` for SAM, `seem_focall_v1.pt` for SEEM) for building our VFMSeg pipeline.

The code of SEEM model is modified to adapt to output pixel-wise logits.

## Scripts
The process of pre-training 2D and 3D networks is similar to xMUDA method, the last checkpoint of those models are utilized for generating pseudo-labels.

test.py 					-- predict pseudo-labels or validate (test) performance.

trial_vfm.py				-- generate VFM guided pseudo-labels.

train_xmuda_with_VFM_mix.py -- training with VFMSeg and FrustrumMixing.

generate_masks_with_VFMs.py -- generate masks beforehand to avoid online calling of VFMs.`pre_defined_proc_list` 	variable defines the dataset to be utilized.

cmd.txt						-- sample commands for initiating training process.

train_baseline.py			-- training script provid by xMUDA method for baseline model.

train_xmuda.py				-- training script provid by xMUDA method for training xMUDA models.and pre-training 2D and 3D models.

## Configurations
..\configs       directory stores configuration files for xMUDA baseline.
..\VFM\configs   directory stores configuration files for our method.

## Main Results
|Ours (VFM-PL + FrustrumMixing)	|   2D	|	3D	 |  Avg.  
| --- | --- | --- | --- 
|A2D2/SemanticKITTI				|  45.0	|  52.3  |	50.0  
|VirualKITTI/SemanticKITTI		|  57.2 |  52.0  |  61.0  
|nuScenesLidarseg:USA/Singapore	|  70.0 |  65.6  |  72.3  
|nuScenesLidarseg:Day/Night		|  60.6 |  70.5  |  66.5  

|xMUDA-PL     					|   2D	|	3D	 |  Avg.  
| --- | --- | --- | --- 
|A2D2/SemanticKITTI				|  41.2	|  49.8  |	47.5  
|VirualKITTI/SemanticKITTI		|  38.7 |  46.1  |  45.0  
|nuScenesLidarseg:USA/Singapore	|  65.6 |  63.8  |  68.4  
|nuScenesLidarseg:Day/Night		|  57.6 |  69.6  |  64.4  

|Oracle      					|   2D	|	3D	 |  Avg.  
| --- | --- | --- | --- 
|A2D2/SemanticKITTI				|  59.3	|  71.9  |	73.6  
|VirualKITTI/SemanticKITTI		|  66.3 |  78.4  |  80.1  
|nuScenesLidarseg:USA/Singapore	|  75.4 |  76.0  |  79.6  
|nuScenesLidarseg:Day/Night		|  61.5 |  69.8  |  69.2  

## xMUDA Code
We would like to thank for authors who kindly share their code:
[Cross-modal Learning for Domain Adaptation in 3D Semantic Segmentation](https://arxiv.org/abs/2101.07253)  
 [Maximilian Jaritz](https://team.inria.fr/rits/membres/maximilian-jaritz/), [Tuan-Hung Vu](https://tuanhungvu.github.io/), [Raoul de Charette](https://team.inria.fr/rits/membres/raoul-de-charette/),  Émilie Wirbel, [Patrick Pérez](https://ptrckprz.github.io/)  


## Preparation
### Prerequisites
Tested with
* PyTorch 1.4
* CUDA 10.0
* Python 3.8
* [SparseConvNet](https://github.com/facebookresearch/SparseConvNet)
* [nuscenes-devkit](https://github.com/nutonomy/nuscenes-devkit)

### Installation
As 3D network we use SparseConvNet. It requires to use CUDA 10.0 (it did not work with 10.1 when we tried).
We advise to create a new conda environment for installation. PyTorch and CUDA can be installed, and SparseConvNet
installed/compiled as follows:
```
$ conda install pytorch=1.4 torchvision cudatoolkit=10.0 -c pytorch
$ pip install --upgrade git+https://github.com/facebookresearch/SparseConvNet.git
```

Clone this repository and install it with pip. It will automatically install the nuscenes-devkit as a dependency.
```
$ git clone https://github.com/valeoai/xmuda.git
$ cd xmuda
$ pip install -ve .
```
The `-e` option means that you can edit the code on the fly.

### Datasets
#### nuScenes-Lidarseg
Please download from the [NuScenes website](https://www.nuscenes.org/nuscenes#download) and extract:
- Full dataset
- nuScenes-lidarseg (All)

You need to perform preprocessing to generate the data for xMUDA first.
The preprocessing subsamples the 360° LiDAR point cloud to only keep the points that project into
the front camera image. All information will be stored in a pickle file (except the images which will be 
read on-the-fly by the dataloader during training).

Please edit the script `xmuda/data/nuscenes_lidarseg/preprocess.py` as follows and then run it.
* `root_dir` should point to the root directory of the NuScenes dataset
* `out_dir` should point to the desired output directory to store the pickle files

#### A2D2
Please download the Semantic Segmentation dataset and Sensor Configuration from the
[Audi website](https://www.a2d2.audi/a2d2/en/download.html) or directly use `wget` and
the following links, then extract.
```
$ wget https://aev-autonomous-driving-dataset.s3.eu-central-1.amazonaws.com/camera_lidar_semantic.tar
$ wget https://aev-autonomous-driving-dataset.s3.eu-central-1.amazonaws.com/cams_lidars.json
```

The dataset directory should have this basic structure:
```
a2d2                                   % A2D2 dataset root
 ├── 20180807_145028
 ├── 20180810_142822
 ├── ...
 ├── cams_lidars.json
 └── class_list.json
```
For preprocessing, we undistort the images and store them separately as .png files.
Similar to NuScenes preprocessing, we save all points that project into the front camera image as well
as the segmentation labels to a pickle file.

Please edit the script `xmuda/data/a2d2/preprocess.py` as follows and then run it.
* `root_dir` should point to the root directory of the A2D2 dataset
* `out_dir` should point to the desired output directory to store the undistorted images and pickle files.
It should be set differently than the `root_dir` to prevent overwriting of images.

#### SemanticKITTI
Please download the files from the [SemanticKITTI website](http://semantic-kitti.org/dataset.html) and
additionally the [color data](http://www.cvlibs.net/download.php?file=data_odometry_color.zip)
from the [Kitti Odometry website](http://www.cvlibs.net/datasets/kitti/eval_odometry.php). Extract
everything into the same folder.

Similar to NuScenes preprocessing, we save all points that project into the front camera image as well
as the segmentation labels to a pickle file.

Please edit the script `xmuda/data/semantic_kitti/preprocess.py` as follows and then run it.
* `root_dir` should point to the root directory of the SemanticKITTI dataset
* `out_dir` should point to the desired output directory to store the pickle files


#### VirtualKITTI

Clone the following repo:
```
$ git clone https://github.com/VisualComputingInstitute/vkitti3D-dataset.git
```
Download raw data and extract with following script:
```
$ cd vkitti3D-dataset/tools
$ mkdir path/to/virtual_kitti
$ bash download_raw_vkitti.sh path/to/virtual_kitti
```
Generate point clouds (npy files):
```
$ cd vkitti3D-dataset/tools
$ for i in 0001 0002 0006 0018 0020; do python create_npy.py --root_path path/to/virtual_kitti --out_path path/to/virtual_kitti/vkitti_npy --sequence $i; done
```

Similar to NuScenes preprocessing, we save all points and segmentation labels to a pickle file.

Please edit the script `xmuda/data/virtual_kitti/preprocess.py` as follows and then run it.
* `root_dir` should point to the root directory of the VirtualKITTI dataset
* `out_dir` should point to the desired output directory to store the pickle files


## UDA Experiments

### xMUDA
You can run the training with
```
$ cd <root dir of this repo>
$ python xmuda/train_xmuda.py --cfg=configs/nuscenes_lidarseg/usa_singapore/uda/xmuda.yaml
```

The output will be written to `/home/<user>/workspace/outputs/xmuda_journal/<config_path>` by 
default. The `OUTPUT_DIR` can be modified in the config file in
(e.g. `configs/nuscenes/usa_singapore/xmuda.yaml`) or optionally at run time in the
command line (dominates over config file). Note that `@` in the following example will be
automatically replaced with the config path, i.e. with `nuscenes_lidarseg/usa_singapore/uda/xmuda`.
```
$ python xmuda/train_xmuda.py --cfg=configs/nuscenes_lidarseg/usa_singapore/uda/xmuda.yaml OUTPUT_DIR path/to/output/directory/@
```

You can start the trainings on the other UDA scenarios (Day/Night and A2D2/SemanticKITTI) analogously:
```
$ python xmuda/train_xmuda.py --cfg=configs/nuscenes_lidarseg/day_night/uda/xmuda.yaml
$ python xmuda/train_xmuda.py --cfg=configs/a2d2_semantic_kitti/uda/xmuda.yaml
$ python xmuda/train_xmuda.py --cfg=configs/virtual_kitti_semantic_kitti/uda/xmuda.yaml
```

### xMUDA<sub>PL</sub>
After having trained the xMUDA model, generate the pseudo-labels as follows:
```
$ python xmuda/test.py --cfg=configs/nuscenes_lidarseg/usa_singapore/uda/xmuda.yaml --pselab @/model_2d_100000.pth @/model_3d_100000.pth DATASET_TARGET.TEST "('train_singapore',)"
```
Note that we use the last model at 100,000 steps to exclude supervision from the validation set by picking the best
weights. The pseudo labels and maximum probabilities are saved as `.npy` file.

Please edit the `pselab_paths` in the config file, e.g. `configs/nuscenes_lidarseg/usa_singapore/uda/xmuda.yaml`,
to match your path of the generated pseudo-labels.

Then start the training. The pseudo-label refinement (discard less confident pseudo-labels) is done
when the dataloader is initialized.
```
$ python xmuda/train_xmuda.py --cfg=configs/nuscenes_lidarseg/usa_singapore/uda/xmuda_pl.yaml
```

You can start the trainings on the other UDA scenarios (Day/Night, A2D2/SemanticKITTI, VirtualKITTI/SemanticKITTI) analogously:
```
$ python xmuda/test.py --cfg=configs/nuscenes_lidarseg/day_night/uda/xmuda.yaml --pselab @/model_2d_100000.pth @/model_3d_100000.pth DATASET_TARGET.TEST "('train_night',)"
$ python xmuda/train_xmuda.py --cfg=configs/nuscenes_lidarseg/day_night/uda/xmuda_pl.yaml

# use batch size 1, because of different image sizes in SemanticKITTI
$ python xmuda/test.py --cfg=configs/a2d2_semantic_kitti/uda/xmuda.yaml --pselab @/model_2d_100000.pth @/model_3d_100000.pth DATASET_TARGET.TEST "('train',)" VAL.BATCH_SIZE 1
$ python xmuda/train_xmuda.py --cfg=configs/a2d2_semantic_kitti/uda/xmuda_pl.yaml

# use batch size 1, because of different image sizes in SemanticKITTI
$ python xmuda/test.py --cfg=configs/virtual_kitti_semantic_kitti/uda/xmuda.yaml --pselab @/model_2d_030000.pth @/model_3d_030000.pth DATASET_TARGET.TEST "('train',)" VAL.BATCH_SIZE 1
$ python xmuda/train_xmuda.py --cfg=configs/nuscenes_lidarseg/day_night/uda/xmuda_pl.yaml
```

### Baseline
Train the baselines (only on source) with:
```
$ python xmuda/train_baseline.py --cfg=configs/nuscenes_lidarseg/usa_singapore/uda/baseline.yaml
$ python xmuda/train_baseline.py --cfg=configs/nuscenes_lidarseg/day_night/uda/baseline.yaml
$ python xmuda/train_baseline.py --cfg=configs/a2d2_semantic_kitti/uda/baseline.yaml
$ python xmuda/train_baseline.py --cfg=configs/virtual_kitti_semantic_kitti/uda/baseline.yaml
```

Non-UDA baseline, training on (labeled) source and labeled target.
```
$ python xmuda/train_baseline_src_trg.py --cfg=configs/nuscenes_lidarseg/usa_singapore/ssda/ssda_baseline_src_trg.yaml
$ python xmuda/train_baseline_src_trg.py --cfg=configs/a2d2_semantic_kitti/ssda/ssda_baseline_src_trg.yaml
$ python xmuda/train_baseline_src_trg.py --cfg=configs/virtual_kitti_semantic_kitti/ssda/ssda_baseline_src_trg.yaml
```

## Testing
You can provide which checkpoints you want to use for testing. We used the ones
that performed best on the validation set during training (the best val iteration for 2D and 3D is
shown at the end of each training). Note that `@` will be replaced
by the output directory for that config file. For example:
```
$ cd <root dir of this repo>
$ python xmuda/test.py --cfg=configs/nuscenes_lidarseg/usa_singapore/uda/xmuda.yaml @/model_2d_065000.pth @/model_3d_095000.pth
```

## License
xMUDA and VFMSeg are released under the [Apache 2.0 license](./LICENSE).
