# calvis
CALVIS: Chest, wAist and peLVIS circumference from 3D human Body meshes for Deep Learning, RFMI 2019

[Yansel Gonzalez Tejeda](http://neoglez.xyz) and [Helmut A. Mayer](https://www.cosy.sbg.ac.at/~helmut/helmut.html)

[[Project page - TBD]](http://example.com) [[arXiv]](https://arxiv.org/abs/2003.00834)

<p align="center">
<img src="/img/subjects_2_6_calvis_segmented.png"
</p>

<p align="center">
<img src="/img/axilla_recognition_80_NN.png"
</p>

<p align="center">
<img src="/img/mesh_signature.png"
</p>

<p align="center">
<img src="/img/experiment_1_results.png"
</p>

<p align="center">
<img src="/img/Calvis-Net.png"
</p>

## Contents
* [1. Download CALVIS dataset](https://github.com/neoglez/calvis#1-download-calvis-dataset)
* [2. or Create your own synthetic data](https://github.com/neoglez/calvis#2-create-your-own-synthetic-data)
* [3. Training models](https://github.com/neoglez/calvis#3-training-models)
* [4. Storage info](https://github.com/neoglez/calvis#4-storage-info)
* [Citation](https://github.com/neoglez/calvis#citation)
* [License](https://github.com/neoglez/calvis#license)
* [Acknowledgements](https://github.com/neoglez/calvis#acknowledgements)

## 1. Download CALVIS dataset


You can check [Storage info](https://github.com/neoglez/calvis#4-storage-info) for how much disk space they require and can do partial download.


The structure of the folders is as follows:

``` shell
CALVIS/dataset/
--------------- cmu/  # using MoCap from CMU dataset
---------------------  annotations/ # json annotations with calvis
----------------------------------  female/
----------------------------------  male/
---------------------  human_body_meshes/ # generated meshes
---------------------------------------- female/
---------------------------------------- male/
---------------------  synthetic_images/ # synthetic greyscale images (200x200x1)
---------------------------------------- 200x200/
------------------------------------------------ female/
------------------------------------------------ male/


```

## 2. or Create your own synthetic data
### 2.1. Preparation
#### 2.1.1. SMPL data

a) You need to download SMPL data from http://smpl.is.tue.mpg.de in order to run the synthetic data generation code. Once you agree on SMPL license terms and have access to downloads, you will have the following three files:

```
basicModel_f_lbs_10_207_0_v1.0.0.pkl
basicmodel_m_lbs_10_207_0_v1.0.0.pkl
smpl_data.npz
```

Place these three files under `datageneration/smpl_data` folder.


``` shell
smpl_data/
--------- smpl_data.npz # 2.5GB
 # trans*           [T x 3]     - (T: number of frames in MoCap sequence)
 # pose*            [T x 72]    - SMPL pose parameters (T: number of frames in MoCap sequence)
 # maleshapes       [1700 x 10] - SMPL shape parameters for 1700 male scans
 # femaleshapes     [2103 x 10] - SMPL shape parameters for 2103 female scans 
 # regression_verts [232]
 # joint_regressor  [24 x 232]
```

#### 2.1.2. Background images with Blender

You need to download [Blender](http://download.blender.org/release/) and install scipy package to run the first part of the code. The provided code was tested with [Blender2.78](http://download.blender.org/release/Blender2.78/blender-2.78a-linux-glibc211-x86_64.tar.bz2), which is shipped with its own python executable as well as distutils package. Therefore, it is sufficient to do the following:

#### 2.1.2. VtkPlotter and Trimesh

### 2.2. Annotating with CALVIS

#### 2.1.2. Calculating chest, waist and pelvis circumference 

## 3. Training models

Here, we provide code to train models on the synthetic data to predict body segmentation or depth. You can also find the models pre-trained on synthetic data.

### 3.1. Preparation

#### 3.1.1. Requirements
* Install [pytorch](https://pytorch.org/) with [CUDA](https://developer.nvidia.com/cuda-downloads) support.
* Install [matio](https://github.com/soumith/matio-ffi.torch) by `luarocks install matio`
* Install [OpenCV-Torch](https://github.com/VisionLabs/torch-opencv) by `luarocks install cv`
* Download [CALVIS](https://github.com/neoglez/calvis)

*Tested on Linux (Ubuntu 16.04) with cuda 10.2*

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

| Dataset     | 8 Meshes | 3803 Meshes | 3803 (200x200x1) Synthetic images | Annotations | Total |
| -----------:|---------:|------------:|----------------------------------:|------------:|------:|
| CALVIS      | 3.3MB    | 1.5GB       |   16MB                            | 1.8MB       | 1.6GB |

## Citation
If you use this code, please cite the following:

```
@misc{ygtham2020calvis,
	title={CALVIS: chest, waist and pelvis circumference from 3D human body 
	meshes as ground truth for deep learning},
	author={Gonzalez Tejeda, YAnsel and Mayer, Helmut A.},
	year={2020},
	eprint={2003.00834},
	archivePrefix={arXiv},
	primaryClass={cs.CV},
	note={14 pages, 6 figures. To appear in the Proceedings of the VIII 
	International Workshop on Representation, analysis and recognition of shape 
	and motion FroM Imaging data (RFMI 2019), 11-13 December 2019, Sidi Bou 
	Said, Tunisia},
}
```

## License
Please check the [license terms](https://github.com/neoglez/calvis/blob/master/LICENSE.md) before downloading and/or using the code, the models and the data.

## Acknowledgements
The [SMPL team](https://smpl.is.tue.mpg.de/) for providing us with the learned human body templates and the SMPL code.

The data generation code and this repo structure is heavely inspired by [Gül Varol's](https://www.robots.ox.ac.uk/~gul/) [SURREAL repo](https://github.com/gulvarol/surreal.

The [vtkplotter team](https://github.com/marcomusy/vtkplotter) (specially Marco Musy) and the [trimesh team](https://github.com/mikedh/trimesh) for the great visualization and intersection libraries.
