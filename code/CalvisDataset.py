# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 10:16:19 2018

@author: yansel
"""

from __future__ import print_function, division
import os
import numpy as np
import json
from torch.utils.data import Dataset
from skimage import io
import matplotlib.pyplot as plt
import random


def read_meshes_from_directory(dir):
    meshes = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            meshes.append(fname)

    return meshes


def read_images_from_directory(dir):
    images = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            images.append(fname)

    return images


def load_annotation_data(data_file_path):
    with open(data_file_path, "r") as data_file:
        return json.load(data_file)


def read_from_obj_file(inmesh_path):
    resolution = 6890

    vertices = np.zeros([resolution, 3], dtype=float)
    faces = np.zeros([resolution - 1, 3], dtype=int)

    with open(inmesh_path, "r") as fp:
        meshdata = np.genfromtxt(fp, usecols=(1, 2, 3))
        vertices = meshdata[:resolution, :]
        faces = meshdata[resolution:, :].astype("int")
    return {"vertices": vertices, "faces": faces}


###############################################################################
#        Human Body Dimensions basic dataset. Defines basic init. logic.
###############################################################################


class CalvisBasic(Dataset):
    """
    Calvis basic dataset.
    It just defines basic initialization logic.
    """

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory containing 2D data and annotation.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.female_root_dir = os.path.join(self.root_dir, "female")
        self.male_root_dir = os.path.join(self.root_dir, "male")
        self.annotation_dir = os.path.join(self.root_dir, "annotations")
        self.female_annotation_dir = os.path.join(
            self.annotation_dir, "female"
        )
        self.male_annotation_dir = os.path.join(self.annotation_dir, "male")


###############################################################################
#        Calvis CMU Dataset.
###############################################################################


class CalvisCMU2DDataset(CalvisBasic):
    """
    CMU Dataset: 2D 200x200 images synthesized with blender plus
    annotations generated automatically by Calvis.
    """

    def __init__(self, root_dir, image_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the 3D body models(meshes)
            and the annotations.
            png_dir (string): Directory with all images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super().__init__(root_dir, transform)
        self.image_dir = image_dir
        self.female_image_dir = os.path.join(self.image_dir, "female")
        self.male_image_dir = os.path.join(self.image_dir, "male")

        for directory in [
            self.root_dir,
            self.female_image_dir,
            self.male_image_dir,
            self.female_annotation_dir,
            self.male_annotation_dir,
        ]:
            if not os.path.isdir(directory):
                raise ValueError(directory + " does not exist!")
        # The dataset is the concatenated images found on female and male
        # directories. The list contains the images filenames.
        images = []
        images_genders = []
        fimages = read_images_from_directory(self.female_image_dir)
        mimages = read_images_from_directory(self.male_image_dir)
        for img in fimages:
            images.append(img)
            images_genders.append(0)
        for img in mimages:
            images.append(img)
            images_genders.append(1)

        self.images = images
        self.images_genders = images_genders
        self.targets = [
            s.split("_mesh_")[0]
            + "_mesh_"
            + s.split("_mesh_")[1][:-4]
            + "_anno.json"
            for s in self.images
        ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        imgfile = ""
        annofile = ""

        if self.images_genders[idx] == 0:
            imgfile = os.path.join(self.female_image_dir, self.images[idx])
            annofile = os.path.join(
                self.female_annotation_dir, self.targets[idx]
            )
        else:
            imgfile = os.path.join(self.male_image_dir, self.images[idx])
            annofile = os.path.join(
                self.male_annotation_dir, self.targets[idx]
            )

        image = io.imread(imgfile, as_gray=False)
        sample = {
            "image": image,
            "annotations": load_annotation_data(annofile),
            "imagefile": imgfile,
            "annotation_file": annofile,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def getFemaleIndxs(self):
        return [
            i
            for i, images_gender in enumerate(self.images_genders)
            if images_gender == 0
        ]

    def getMaleIndxs(self):
        return [
            i
            for i, images_gender in enumerate(self.images_genders)
            if images_gender == 1
        ]


###############################################################################
class CalvisFairCMU2DDataset(CalvisCMU2DDataset):
    """
    Returns a dataset that contains all male instances (1700) plus the same
    amount of female instances. Since there are more female instances than
    male, female instances are random selected. The generator can be optionally
    seed.
    """

    def __init__(self, root_dir, image_dir, transform=None, seed=None):
        """
        Args:
            root_dir (string): Directory with all the 3D body models(meshes)
            and the annotations.
            png_dir (string): Directory with all images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super().__init__(root_dir, image_dir, transform)
        if seed is not None:
            random.seed(seed)

        male_instances_count = 1700

        female_indxs = self.getFemaleIndxs()
        male_indxs = self.getMaleIndxs()
        female_indxs = random.sample(female_indxs, male_instances_count)
        samples = female_indxs + male_indxs

        self.images = [self.images[idx] for idx in samples]
        self.images_genders = [self.images_genders[idx] for idx in samples]
        self.targets = [
            s.split("_mesh_")[0]
            + "_mesh_"
            + s.split("_mesh_")[1][:-4]
            + "_anno.json"
            for s in self.images
        ]


###############################################################################
# Calvis Dataset for skorch. Only the needed values are returned              #
###############################################################################
class CalvisForSkorchFairCMU2DDataset(CalvisFairCMU2DDataset):
    """
    Returns a dataset that contains all male instances (1700) plus the same
    amount of female instances. Since there are more female instances than
    male, female instances are random selected. The generator can be optionally
    seed. Only the needed values are returned tomake it compatible with skorch.
    """

    def __init__(self, root_dir, image_dir, transform=None, seed=None):
        """
        Args:
            root_dir (string): Directory with all the 3D body models(meshes)
            and the annotations.
            png_dir (string): Directory with all images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super().__init__(root_dir, image_dir, transform, seed)

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        # get the inputs
        x = data["image"]
        y = data["annotations"]["human_dimensions"][0]
        return (x, y)

###############################################################################


if __name__ == "__main__":

    rootDir = "/home/neoglez/cmu/dataset/"
    imageDir = "/home/neoglez/cmu/dataset/synthetic_images/200x200/"

    train_calvis_cmu_2d_dataset = CalvisCMU2DDataset(
        root_dir=rootDir, image_dir=imageDir
    )

    fair_calvis_cmu_2d_dataset = CalvisFairCMU2DDataset(
        root_dir=rootDir, image_dir=imageDir
    )

    # assert that these two datasets do not have the same length
    assert (
        bool(
            len(train_calvis_cmu_2d_dataset)
            & len(fair_calvis_cmu_2d_dataset)
        )
    ) == True, "Train and Fair datasets have the same lenght!"

    # just testing
    print(train_calvis_cmu_2d_dataset[958])
    print(train_calvis_cmu_2d_dataset.images_genders[3001])

    female_indxs = train_calvis_cmu_2d_dataset.getFemaleIndxs()
    male_indxs = train_calvis_cmu_2d_dataset.getMaleIndxs()

    # chest circumference
    all_cc = []
    # waist circumference
    all_wc = []
    # pelvis circumference
    all_pc = []

    # get 2 males and 2 females randomly
    females = random.sample(female_indxs, 2)
    males = random.sample(male_indxs, 2)
    items_list = np.array(females + males)

    # Show only the pictures
    if True:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        dimensions = []
        for idx, i in enumerate(items_list):
            sample = train_calvis_cmu_2d_dataset[i]
            dimensions = sample["annotations"]["human_dimensions"]

            print(i, sample["image"].shape, dimensions["chest_circumference"])
            print(i, sample["image"].shape, dimensions["waist_circumference"])
            print(i, sample["image"].shape, dimensions["pelvis_circumference"])

            globals()["ax%s" % (idx + 1)].axis("off")
            globals()["ax%s" % (idx + 1)].imshow(sample["image"])
            globals()["ax%s" % (idx + 1)].set_title("Sample #{}".format(i))

            globals()["ax%s" % (idx + 1)].annotate(
                "CC: {0:.2f} m".format(dimensions["chest_circumference"]),
                xy=(0, 100),
                xycoords="axes pixels",
                horizontalalignment="right",
                verticalalignment="top",
            )

        #            globals()['ax%s' % (idx + 1)].annotate('axes fraction',
        #            xy=(3, 100), xycoords='data',
        #            xytext=(0.6, 0.5), textcoords='axes fraction',
        #            horizontalalignment='right', verticalalignment='top')
        #
        #            globals()['ax%s' % (idx + 1)].annotate('axes fraction',
        #            xy=(3, 100), xycoords='data',
        #            xytext=(0.6, 0.5), textcoords='axes fraction',
        #            horizontalalignment='right', verticalalignment='top')
        #            ax.set_title(
        #                    ("Sample #{}, "
        #                     "chest_circumference: {}, "
        #                     "waist_circumference: {}, "
        #                     "pelvis_circumference: {}").format(
        #                         i,
        #                         dimensions['chest_circumference'],
        #                         dimensions['waist_circumference'],
        #                         dimensions['pelvis_circumference'])
        #            )
        # plt.tight_layout()
        plt.show()
