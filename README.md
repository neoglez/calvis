# calvis
CALVIS: Chest, wAist and peLVIS circumference from 3D human Body meshes for Deep Learning, RFMI 2019

[Yansel Gonzalez Tejeda](https://github.com/neoglez) and [Helmut A. Mayer](https://www.cosy.sbg.ac.at/~helmut/helmut.html)

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
* [2. or Create your own synthetic data](https://github.com/neoglez/calvis#2-or-create-your-own-synthetic-data)
* [3. Training models](https://github.com/neoglez/calvis#3-training-models)
* [4. Storage info](https://github.com/neoglez/calvis#4-storage-info)
* [Citation](https://github.com/neoglez/calvis#citation)
* [License](https://github.com/neoglez/calvis#license)
* [Acknowledgements](https://github.com/neoglez/calvis#acknowledgements)

## 1. Download CALVIS dataset


You can check [Storage info](https://github.com/neoglez/calvis#4-storage-info) for how much disk space they require and can do partial download.
Download from our cloud (see bellow). College researchers have asked for an 8-meshes package and a small dataset with 100 intances, we also provide those.
| Dataset  |  Download Link     | sha256sum      |  Password |
|----------|:-------------:|---------------:|---------------:|
| CALVIS (full) |  [CALVIS.tar.gz](https://cloudlogin03.world4you.com/index.php/s/VowlhwRR97y4xjK) | ab5d48c57677a7654c073e3148fc545cb335320961685886ed8ea8fef870b15e   | calvisdataset   |
| Cavis, only 8 human meshes |    [cavis-8-human-meshes.tar.gz](https://cloudlogin03.world4you.com/index.php/s/KC8N9YFKDFUm6Du)   |   8c457ad064829c439b977ebc2e6487e3e5fbb09203d78efd226ab0793081aafd   | calvis-8-hm   |
| Calvis, small dataset with 100 instances | [calvis-100-instances.tar.gz](https://cloudlogin03.world4you.com/index.php/s/F55LZZiQRMer45X) |  2812ea4cba4e521fa0dac96d7a2b2ef063e1099d8a7db09bab4ea285746ad417   | calvis-100-i    |

The general structure of the folders is as follows:

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

Please consider that in all cases, we install dependencies into a conda environment. The code was tested under ubuntu 16.04 with python 3.7.

#### 2.1.1. SMPL data

You need to download SMPL data from http://smpl.is.tue.mpg.de and https://www.di.ens.fr/willow/research/surreal/data/ in order to run the synthetic data generation code. Once you agree on SMPL license terms and have access to downloads, you will have the following three files:

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

#### 2.1.2. Human Body Models utilities

You need to install [Human Body Models](https://github.com/neoglez/hbm). Please, consider installing all dependencies in a conda environment.

``` shell

git clone http://github.com/neoglez/hbm.git
cd hbm
pip install .
```

#### 2.1.3. Synthetic images with Blender

Building Blender is a painful process. That is why we recommend to download and install the version that we used. The provided code was tested with [Blender2.78](http://download.blender.org/release/Blender2.78/blender-2.78a-linux-glibc211-x86_64.tar.bz2).

Just open the Scripting view and load (or copy and paste) the script `synthesize_cmu_200x200_grayscale_images.py`
Change the path correspondingly at `cmu_dataset_path = os.path.abspath("/home/youruser/YourCode/calvis/CALVIS/dataset/cmu/")` and run the script.
The process takes several minutes.

#### 2.1.4. VtkPlotter and Trimesh

You need to install these two libraries:

``` shell

conda install -c conda-forge vtkplotter
pip install trimesh
```

### 2.2. Annotating with CALVIS

#### 2.1.2. Calculating chest, waist and pelvis circumference
Run the script `CalvisToCMUAnnotazer.py`
The process takes several hours.

#### 2.1.3. Visualize chest, waist and pelvis circumference
To visualize at which points calvis is calculating the body measurements, follow the code in `display_one_by_one_8_subjects_calvis_with_vtkplotter_and_trimesh.py` or directly display it with jupyter notebook `display_one_by_one_8_subjects_calvis_with_vtkplotter_and_trimesh.ipynb`

Note: To display the meshes in the browser, we use k3d backend. Install it with

``` shell

conda install -c conda-forge k3d
```

## 3. Training and evaluating CalvisNet

At this point you should have the input (synthetic images) and the supervision signal (calvis annotations). Here, we provide code to train and evaluate CalvisNet on the synthetic data to predict given the input chest, waist and pelvis circumference.

### 3.1. Preparation

#### 3.1.1. Requirements
* Install [pytorch](https://pytorch.org/) with [CUDA](https://developer.nvidia.com/cuda-downloads) support.
* Download [CALVIS](https://github.com/neoglez/calvis)
* Install scikit-learn, SciPy and its image processing routines

``` shell

conda install scikit-learn 
conda install -c anaconda scipy
conda install -c anaconda scikit-image
```

*Tested on Linux (Ubuntu 16.04) with cuda 10.2 on a GeForce GTX 1060 6GB graphic card*
To train and evaluate calvis, follow the code in `train_calvis-net_cross_validation.py`

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
	author={Gonzalez Tejeda, Yansel and Mayer, Helmut A.},
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

The data generation code and this repo structure is heavely inspired by [Gül Varol's](https://www.robots.ox.ac.uk/~gul/) [SURREAL repo](https://github.com/gulvarol/surreal).

The [vtkplotter team](https://github.com/marcomusy/vtkplotter) (specially Marco Musy) and the [trimesh team](https://github.com/mikedh/trimesh) for the great visualization and intersection libraries.
