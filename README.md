# calvis
CALVIS: Chest, wAist and peLVIS circumference from 3D human Body meshes for Deep Learning, RFMI 2019

<p align="center">
<img src="http://example.com/calvis.gif"
</p>


[Yansel Gonzalez Tejeda](http://example.com/ygt/) and [Helmut A. Mayer](https://www.cosy.sbg.ac.at/~helmut/helmut.html)

[[Project page - TBD]](http://example.com) [[arXiv]](https://arxiv.org/abs/2003.00834)

<p align="center">
<img src="http://www.di.ens.fr/willow/research/surreal/images/surreal.gif"
</p>

## Contents
* [1. Download CALVIS dataset](https://github.com/neoglez/surreal#1-download-surreal-dataset)
* [2. Create your own synthetic data](https://github.com/neoglez/calvis#2-create-your-own-synthetic-data)
* [3. Training models](https://github.com/neoglez/calvis#3-training-models)
* [4. Storage info](https://github.com/neoglez/calvis#4-storage-info)
* [Citation](https://github.com/neoglez/surreal#citation)
* [License](https://github.com/neoglez/calvis#license)
* [Acknowledgements](https://github.com/neoglez/calvis#acknowledgements)

## 1. Download CALVIS dataset

In order to download SURREAL dataset, you need to accept the license terms. The links to license terms and download procedure are available here:

https://www.di.ens.fr/willow/research/surreal/data/

Once you receive the credentials to download the dataset, you will have a personal username and password. Use these either to download the dataset excluding optical flow data from [here: (SURREAL_v1.tar.gz, 86GB)](https://lsh.paris.inria.fr/SURREAL/SURREAL_v1.tar.gz) or download individual files with the `download/download_surreal.sh` script as follows:

``` shell
./download_surreal.sh /path/to/dataset yourusername yourpassword
```

You can check [Storage info](https://github.com/gulvarol/surreal#4-storage-info) for how much disk space they require and can do partial download.

Find under `datageneration/misc/3Dto2D` scripts that explain the projective relations between joints2D and joints3D variables.

The structure of the folders is as follows:

``` shell
SURREAL/data/
------------- cmu/  # using MoCap from CMU dataset
-------------------- train/
-------------------- val/ # small subset of test 
-------------------- test/
----------------------------  run0/ #50% overlap
----------------------------  run1/ #30% overlap
----------------------------  run2/ #70% overlap
------------------------------------  <sequenceName>/ #e.g. 01_01
--------------------------------------------------  <sequenceName>_c%04d.mp4        # RGB - 240x320 resolution video
--------------------------------------------------  <sequenceName>_c%04d_depth.mat  # Depth
#     depth_1,   depth_2, ...  depth_T [240x320 single] - in meters
--------------------------------------------------  <sequenceName>_c%04d_segm.mat   # Segmentation
#     segm_1,     segm_2, ...   segm_T [240x320 uint8]  - 0 for background and 1..24 for SMPL body parts
--------------------------------------------------  <sequenceName>_c%04d_gtflow.mat # Ground truth optical flow
#     gtflow_1, gtflow_2, ... gtflow_T [240x320x2 single]
--------------------------------------------------  <sequenceName>_c%04d_info.mat   # Remaining annotation
#     bg           [1xT cell]      - names of background image files
#     camDist      [1 single]      - camera distance
#     camLoc       [3x1 single]    - camera location
#     clipNo       [1 double]      - clip number of the full sequence (corresponds to the c%04d part of the file)
#     cloth        [1xT cell]      - names of texture image files
#     gender       [Tx1 uint8]     - gender (0: 'female', 1: 'male')
#     joints2D     [2x24xT single] - 2D coordinates of 24 SMPL body joints on the image pixels
#     joints3D     [3x24xT single] - 3D coordinates of 24 SMPL body joints in real world meters
#     light        [9x100 single]  - spherical harmonics lighting coefficients
#     pose         [72xT single]   - SMPL parameters (axis-angle)
#     sequence     [char]          - <sequenceName>_c%04d
#     shape        [10xT single]   - body shape parameters
#     source       [char]          - 'cmu'
#     stride       [1 uint8]       - percent overlap between clips, 30 or 50 or 70
#     zrot         [Tx1 single]    - rotation in Z (euler angle)

# *** T is the number of frames, mostly 100.

```

*Note: There are some monster shapes in the dataset which were not cleaned before training. Some subjects spotted by visual inspection are `18`, `19`, `143_21`.*

## 2. Create your own synthetic data
### 2.1. Preparation
#### 2.1.1. SMPL data

a) You need to download SMPL for MAYA from http://smpl.is.tue.mpg.de in order to run the synthetic data generation code. Once you agree on SMPL license terms and have access to downloads, you will have the following two files:

```
basicModel_f_lbs_10_207_0_v1.0.2.fbx
basicModel_m_lbs_10_207_0_v1.0.2.fbx
```

Place these two files under `datageneration/smpl_data` folder.

b) With the same credentials as with the SURREAL dataset, you can download the remaining necessary SMPL data and place it in `datageneration/smpl_data`.

``` shell
./download_smpl_data.sh /path/to/smpl_data yourusername yourpassword
```

``` shell
smpl_data/
------------- textures/ # folder containing clothing images (also available at lsh.paris.inria.fr/SURREAL/smpl_data/textures.tar.gz)
------------- (fe)male_beta_stds.npy
------------- smpl_data.npz # 2.5GB
 # trans*           [T x 3]     - (T: number of frames in MoCap sequence)
 # pose*            [T x 72]    - SMPL pose parameters (T: number of frames in MoCap sequence)
 # maleshapes       [1700 x 10] - SMPL shape parameters for 1700 male scans
 # femaleshapes     [2103 x 10] - SMPL shape parameters for 2103 female scans 
 # regression_verts [232]
 # joint_regressor  [24 x 232]
```

*Note: SMPL pose parameters are [MoSh](http://mosh.is.tue.mpg.de/)'ed from CMU MoCap data. Note that these are not the most recent MoSh results. For any questions regarding MoSh, please contact mosh@tue.mpg.de instead. Here, we only provide the pose parameters for MoCap sequences, not their shape parameters (they are not used in this work, we randomly sample body shapes).*

#### 2.1.2. Background images


#### 2.1.3. Blender
You need to download [Blender](http://download.blender.org/release/) and install scipy package to run the first part of the code. The provided code was tested with [Blender2.78](http://download.blender.org/release/Blender2.78/blender-2.78a-linux-glibc211-x86_64.tar.bz2), which is shipped with its own python executable as well as distutils package. Therefore, it is sufficient to do the following:

### 2.2. Running the code

## 3. Training models

Here, we provide code to train models on the synthetic data to predict body segmentation or depth. You can also find the models pre-trained on synthetic data.

### 3.1. Preparation

#### 3.1.1. Requirements
* Install [Torch](https://github.com/torch/distro) with [cuDNN](https://developer.nvidia.com/cudnn) support.
* Install [matio](https://github.com/soumith/matio-ffi.torch) by `luarocks install matio`
* Install [OpenCV-Torch](https://github.com/VisionLabs/torch-opencv) by `luarocks install cv`
* Download [CALVIS](https://github.com/neoglez/calvis)

*Tested on Linux (Ubuntu 16.04) with cuda v8 and cudNN v5.1. Let me know if there are other major dependencies that I forgot to include.*

#### 3.1.2. Setup paths
Place the data under `~/datasets/CALVIS` or change the `opt.dataRoot` in opts.lua. The outputs will be written to `~/cnn_saves/<datasetname>/<experiment>`, you can change the `opt.logRoot` to change the `cnn_saves` location.

### 3.2. Running the code

#### 3.2.1. Train
There are sample scripts under `training/exp/train` directory that are self-explanatory. Those are used for the 'Synth' experiments in the paper. Check `opts.lua` script to see what options are available.

#### 3.2.2. Visualize
A few display functionalities are implemented to debug and visualize results. Example usage:
```
./training/exp/vis.sh 1 30 cmu eval val
```

#### 3.2.3. Evaluate
To obtain the final results, you can run x.x.


## 4. Storage info

You might want to do a partial download depending on your needs.

| Dataset            | 1    |  2    | Total  |
| ------------------ |-----:| -----:| ------:|
| **CALVIS**         | 3.8G | 6.0G  | 82.5G  |

## Citation
If you use this code, please cite the following:

```
@misc{tejeda2020calvis,
    title={CALVIS: chest, waist and pelvis circumference from 3D human body meshes as ground truth for deep learning},
    author={Gonzalez Tejeda, YAnsel and Mayer, Helmut A.},
    year={2020},
    eprint={2003.00834},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    note={14 pages, 6 figures. To appear in the Proceedings of the VIII International Workshop on Representation, analysis and recognition of shape and motion FroM Imaging data (RFMI 2019), 11-13 December 2019, Sidi Bou Said, Tunisia},
}
```

## License
Please check the [license terms](https://github.com/neoglez/calvis/blob/master/LICENSE.md) before downloading and/or using the code, the models and the data.

## Acknowledgements
The data generation code is built by YGT

The training code is written by YGT
