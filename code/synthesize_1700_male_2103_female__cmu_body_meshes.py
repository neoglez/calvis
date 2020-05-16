#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 09:22:49 2019

@author: neoglez
"""
import numpy as np
import pickle
import hbm
import os
import json
import math

cmu_dataset_path = "./CALVIS/dataset/cmu/"
cmu_dataset_meshes_path = "./CALVIS/dataset/cmu/human_body_meshes/"
cmu_dataset_meshes_path_length = len(cmu_dataset_meshes_path)
cmu_dataset_annotation_path = "./CALVIS/dataset/cmu/annotations/"

smpl_data_folder = "../datageneration/smpl_data/"

SMPL_basicModel_f_lbs_path = (
    smpl_data_folder + "basicModel_f_lbs_10_207_0_v1.0.0.pkl"
)
SMPL_basicModel_m_lbs_path = (
    smpl_data_folder + "basicmodel_m_lbs_10_207_0_v1.0.0.pkl"
)

smpl_data_filename = "smpl_data.npz"

smpl_data = np.load(os.path.join(smpl_data_folder, smpl_data_filename))

maleshapes = smpl_data["maleshapes"]
femaleshapes = smpl_data["femaleshapes"]


betas = {"female": femaleshapes, "male": maleshapes}

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

new_female_joint_regressor = hbm.KJointPredictor(k_joints_predictor)

k_joints_predictor = male_model.get("J_regressor").A

new_male_joint_regressor = hbm.KJointPredictor(k_joints_predictor)

####################################################################
# Initialize the Osmpl female and male template.                   #
####################################################################
new_female_template = hbm.OSmplTemplate(
    female_model.get("v_template"),
    female_model.get("f"),
    female_model.get("blend_weights"),
    female_model.get("shape_blend_shapes"),
    new_female_joint_regressor,
    female_model.get("posedirs"),
    None
)
new_male_template = hbm.OSmplTemplate(
    male_model.get("v_template"),
    male_model.get("f"),
    male_model.get("blend_weights"),
    male_model.get("shape_blend_shapes"),
    new_male_joint_regressor,
    male_model.get("posedirs"),
    None
)

####################################################################
# Once we have the template we instanciate the complete model.     #
####################################################################
human_female_model = hbm.OSmplWithPose(
    new_female_template, female_model.get("shapedirs").x, None, None
)
human_male_model = hbm.OSmplWithPose(
    new_male_template, male_model.get("shapedirs").x, None, None
)

# Number of PCA components: The shapedirs is a tensor of shape
# number_of_vertices x number_of_vertex_coordinate x number_of_PCA.
# In our case this is 6890 x 3 x 10 for both female and male models.
number_of_PCAs = female_model.get("shapedirs").shape[-1]

synthesizer = hbm.Synthesizer(
    "smpl",
    number_of_male_models=1,
    number_of_female_models=1,
    smpl_female_model=human_female_model,
    smpl_male_model=human_male_model,
    number_of_PCAs=number_of_PCAs,
    hbm_path=cmu_dataset_path,
)

already_synthesized_females = 0
already_synthesized_males = 0
already_synthesized = 0

padding_f = int(math.log10(len(betas["female"]))) + 1
padding_m = int(math.log10(len(betas["male"]))) + 1
padding = None

for gender in ["female", "male"]:
    for i, beta in enumerate(betas[gender], 1):
        synthesizer.synthesize_human(beta, gender)
        if gender == "female":
            padding = padding_f
            already_synthesized_females += 1
            already_synthesized = already_synthesized_females
        else:
            padding = padding_m
            already_synthesized_males += 1
            already_synthesized = already_synthesized_males

        # the name/path for this mesh
        outmesh_path = (
            cmu_dataset_meshes_path
            + gender
            + "/"
            + "subject_mesh_%0.*d.obj" % (padding, already_synthesized)
        )
        annotations_path = (
            cmu_dataset_annotation_path
            + outmesh_path[cmu_dataset_meshes_path_length:-4]
            + "_anno.json"
        )

        synthesizer.save_human_mesh(gender, outmesh_path)

        with open(annotations_path, "w") as fp:
            # Write the betas. Since "the betas" is a matrix, we have to
            # 'listifyit'.
            json.dump(
                {"betas": beta.tolist(), "human_dimensions": {}},
                fp,
                sort_keys=True,
                indent=4,
                ensure_ascii=False,
            )
