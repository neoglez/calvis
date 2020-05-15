#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 09:22:49 2019

@author: neoglez
"""
import numpy as np
import pickle
from OSmpl import KJointPredictor, OSmplTemplate, OSmpl
from Synthesizer import Synthesizer

SMPL_basicModel_f_lbs_path = "./basicModel_f_lbs_10_207_0_v1.0.0.pkl"
SMPL_basicModel_m_lbs_path = "./basicmodel_m_lbs_10_207_0_v1.0.0.pkl"

betas = np.array(
    [
        np.array(
            [
                2.25176191,
                -3.7883464,
                0.46747496,
                3.89178988,
                2.20098416,
                0.26102114,
                -3.07428093,
                0.55708514,
                -3.94442258,
                -2.88552087,
            ]
        ),  # fat
        np.array(
            [
                -2.26781107,
                0.88158132,
                -0.93788176,
                -0.23480508,
                1.17088298,
                1.55550789,
                0.44383225,
                0.37688275,
                -0.27983086,
                1.77102953,
            ]
        ),  # thin
        np.array(
            [
                0.00404852,
                0.8084637,
                0.32332591,
                -1.33163664,
                1.05008727,
                1.60955275,
                0.22372946,
                -0.10738459,
                0.89456312,
                -1.22231216,
            ]
        ),  # short
        np.array(
            [
                3.63453289,
                1.20836171,
                3.15674431,
                -0.78646793,
                -1.93847355,
                -0.32129994,
                -0.97771656,
                0.94531640,
                0.52825811,
                -0.99324327,
            ]
        ),  # tall
    ]
)

SMPL_basicModel_f_lbs_path = (
    "/media/neoglez/Data1/privat/PhD_Uni_Salzburg"
    "/DATASETS/smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl"
)
SMPL_basicModel_m_lbs_path = (
    "/media/neoglez/Data1/privat/PhD_Uni_Salzburg"
    "/DATASETS/smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl"
)

try:
    # Load pkl created in python 2.x with python 2.x
    female_model = pickle.load(open(SMPL_basicModel_f_lbs_path, "rb"))
    male_model = pickle.load(open(SMPL_basicModel_m_lbs_path, "rb"))
except:
    # Load pkl created in python 2.x with python 3.x
    female_model = pickle.load(
        open(SMPL_basicModel_f_lbs_path, "rb"), encoding="latin1"
    )
    male_model = pickle.load(
        open(SMPL_basicModel_m_lbs_path, "rb"), encoding="latin1"
    )

####################################################################
# Initialize the joints regressor as dense array (for clarity).    #
####################################################################

k_joints_predictor = female_model.get("J_regressor").A

new_female_joint_regressor = KJointPredictor(k_joints_predictor)

k_joints_predictor = male_model.get("J_regressor").A

new_male_joint_regressor = KJointPredictor(k_joints_predictor)

####################################################################
# Initialize the Osmpl female and male template.                   #
####################################################################
new_female_template = OSmplTemplate(
    female_model.get("v_template"),
    female_model.get("f"),
    female_model.get("blend_weights"),
    female_model.get("shape_blend_shapes"),
    new_female_joint_regressor,
    female_model.get("posedirs"),
)
new_male_template = OSmplTemplate(
    male_model.get("v_template"),
    male_model.get("f"),
    male_model.get("blend_weights"),
    male_model.get("shape_blend_shapes"),
    new_male_joint_regressor,
    male_model.get("posedirs"),
)

####################################################################
# Once we have the template we instanciate the complete model.     #
####################################################################
human_female_model = OSmpl(
    new_female_template, female_model.get("shapedirs").x, None, None
)
human_male_model = OSmpl(
    new_male_template, male_model.get("shapedirs").x, None, None
)

# Number of PCA components: The shapedirs is a tensor of shape
# number_of_vertices x number_of_vertex_coordinate x number_of_PCA.
# In our case this is 6890 x 3 x 10 for both female and male models.
number_of_PCAs = female_model.get("shapedirs").shape[-1]

hbm_dataset_path = "/home/neoglez/calvis/data/human_body_meshes/"
# hbm_dataset_path = 'c:/Users/yansel/Documents/privat/H_DIM_Project/'

synthesizer = Synthesizer(
    "smpl",
    number_of_male_models=1,
    number_of_female_models=1,
    smpl_female_model=human_female_model,
    smpl_male_model=human_male_model,
    number_of_PCAs=number_of_PCAs,
    hbm_path=hbm_dataset_path,
)

for gender in ["female", "male"]:
    for i, beta in enumerate(betas, 1):
        synthesizer.synthesize_human(beta, gender)
        # the name/path for this mesh
        outmesh_path = "subject_mesh_%0.*d.obj" % (2, i)
        outmesh_path = hbm_dataset_path + gender + "/" + outmesh_path
        synthesizer.save_human_mesh(gender, outmesh_path)
