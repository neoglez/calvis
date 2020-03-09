#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 10:18:17 2020

@author: neoglez
"""
import sys

hbm_path = "/home/neoglez/hbm/"

sys.path.append(hbm_path)

import numpy as np
import pickle
from OSmpl import KJointPredictor, OSmplTemplate, OSmpl
from Synthesizer import Synthesizer
import os
import json
import math
import locale
import trimesh
#locale.setlocale(locale.LC_NUMERIC,"C")

cmu_dataset_path = '/home/neoglez/cmu'
cmu_dataset_meshes_path = '/home/neoglez/cmu/dataset/human_body_meshes/'
cmu_dataset_meshes_path_length = len(cmu_dataset_meshes_path)
cmu_dataset_annotation_path = '/home/neoglez/cmu/dataset/annotations/'

SMPL_basicModel_f_lbs_path = "./basicModel_f_lbs_10_207_0_v1.0.0.pkl"
SMPL_basicModel_m_lbs_path = "./basicmodel_m_lbs_10_207_0_v1.0.0.pkl"

smpl_data_folder = ("/home/neoglez/smpl_data/SURREAL/smpl_data/")

smpl_data_filename = ("smpl_data.npz")

smpl_data = np.load(os.path.join(smpl_data_folder, smpl_data_filename))

maleshapes = smpl_data['maleshapes']
femaleshapes = smpl_data['femaleshapes']


betas = {'female': femaleshapes, 'male': maleshapes}

SMPL_basicModel_f_lbs_path = ("/media/neoglez/Data1/privat/PhD_Uni_Salzburg"
              "/DATASETS/smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl")
SMPL_basicModel_m_lbs_path = ("/media/neoglez/Data1/privat/PhD_Uni_Salzburg"
              "/DATASETS/smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl")

try:
    # Load pkl created in python 2.x with python 2.x
    female_model = pickle.load(open(SMPL_basicModel_f_lbs_path, 'rb'))
    male_model = pickle.load(open(SMPL_basicModel_m_lbs_path, 'rb'))
except:
    # Load pkl created in python 2.x with python 3.x
    female_model = pickle.load(open(SMPL_basicModel_f_lbs_path, 'rb'),
                               encoding='latin1')
    male_model = pickle.load(open(SMPL_basicModel_m_lbs_path, 'rb'),
                             encoding='latin1')

    
# print some betas
#print(betas['female'][2])
#print(betas['male'][2])

####################################################################
# Initialize the joints regressor as dense array (for clarity).    #
####################################################################

k_joints_predictor = female_model.get('J_regressor').A

new_female_joint_regressor = KJointPredictor(k_joints_predictor)

k_joints_predictor = male_model.get('J_regressor').A

new_male_joint_regressor = KJointPredictor(k_joints_predictor)

####################################################################
# Initialize the Osmpl female and male template.                   #
####################################################################
new_female_template = OSmplTemplate(female_model.get('v_template'),
                                    female_model.get('f'),
                                    female_model.get('blend_weights'),
                                    female_model.get('shape_blend_shapes'),
                                    new_female_joint_regressor, 
                                    female_model.get('posedirs'))
new_male_template = OSmplTemplate(male_model.get('v_template'),
                                    male_model.get('f'),
                                    male_model.get('blend_weights'),
                                    male_model.get('shape_blend_shapes'),
                                    new_male_joint_regressor, 
                                    male_model.get('posedirs'))

####################################################################
# Once we have the template we instanciate the complete model.     #
####################################################################
human_female_model = OSmpl(new_female_template,
                           female_model.get('shapedirs').x,
                           None, None)
human_male_model = OSmpl(new_male_template,
                         male_model.get('shapedirs').x,
                           None, None)

# Number of PCA components: The shapedirs is a tensor of shape
# number_of_vertices x number_of_vertex_coordinate x number_of_PCA.
# In our case this is 6890 x 3 x 10 for both female and male models.
number_of_PCAs = female_model.get('shapedirs').shape[-1]

synthesizer = Synthesizer('smpl', number_of_male_models=1,
                          number_of_female_models=1,
                          smpl_female_model=human_female_model,
                          smpl_male_model=human_male_model,
                          number_of_PCAs = number_of_PCAs,
                          hbm_path = cmu_dataset_path)

already_synthesized_females = 0
already_synthesized_males = 0
already_synthesized = 0

padding_f = int(math.log10(len(betas['female']))) +1
padding_m = int(math.log10(len(betas['male']))) +1
padding = None

import random
# @todo implement a button to select gender
# select a random gender
gender = random.choice(['female', 'male'])

# manually set
#gender = 'female'
#print(gender)
#print(len(betas[gender]))

# @todo implement sliders for the 10 principal components
# select a random element
random_betas_idx = random.randint(0, len(betas[gender]) - 1)
#print(random_betas_idx)
betas = betas[gender][random_betas_idx]
#print(betas)

# @todo Should we save and then show/plot?
# Synthesize human
synthesizer.synthesize_human(betas, gender)
# write to file
#human = './test_human.obj'
#synthesizer.save_human_mesh(gender, human)
human = synthesizer.return_as_obj_format(gender)

from vtkplotter import Plotter, Mesh, settings, Points
verts = np.array([line.split()[1:] for line in human.split('\n') if line.startswith('v ')], dtype="f8")
#print(verts[0])
faces = np.array([line.split()[1:] for line in human.split('\n') if line.startswith('f ')], dtype="int")
# remember that faces are 1-indexed in obj files therefore we have to
# one from every element.
faces = faces - 1
#faces = np.array([])
#print(faces[0])

# the way vertices are assembled into polygons can be retrieved
# in two different formats:
#printc('points():\n', m.points()[0])
#printc('faces(): \n', m.faces()[0])

# attach to logger so trimesh messages will be printed to console
trimesh.util.attach_to_log()

# mesh objects can be created from existing faces and vertex data
mesh = trimesh.Trimesh(vertices=verts,
                       faces=faces)
#print(mesh.is_watertight)
mesh.show()
