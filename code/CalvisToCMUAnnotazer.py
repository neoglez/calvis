# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 09:47:47 2018

@author: yansel
"""
import numpy as np
import hbm
import os
import locale
from calvis import Calvis
import time
import json

locale.setlocale(locale.LC_NUMERIC, "C")

cmu_dataset_path = os.path.abspath("./../CALVIS/dataset/cmu/")
cmu_dataset_meshes_path = os.path.join(cmu_dataset_path, "human_body_meshes/")


male_directory = os.path.join(cmu_dataset_meshes_path, "male")
png_male_path = os.path.join(cmu_dataset_path, "synthetic_images/200x200/male")
female_directory = os.path.join(cmu_dataset_meshes_path, "female")
png_female_path = os.path.join(
    cmu_dataset_path, "synthetic_images/200x200/female"
)

cmu_dataset_meshes_path_length = len(cmu_dataset_meshes_path)
cmu_dataset_annotation_path = os.path.join(
    cmu_dataset_path, "annotations/"
)


calvis = Calvis()

files = []
genders = []
meshes = []
# r=root, d=directories, f = files
for r, d, f in os.walk(cmu_dataset_meshes_path):
    for file in f:
        if ".obj" in file:
            files.append(os.path.join(r, file))
            genders.append("female" if "female" == r[-6:] else "male")

smpl_data_folder = os.path.abspath("./../datageneration/smpl_data")

SMPL_basicModel_f_lbs_path = os.path.join(
    smpl_data_folder, "basicModel_f_lbs_10_207_0_v1.0.0.pkl"
)
SMPL_basicModel_m_lbs_path = os.path.join(
    smpl_data_folder, "basicmodel_m_lbs_10_207_0_v1.0.0.pkl"
)


SMPL_basicModel = {
    "f": SMPL_basicModel_f_lbs_path,
    "m": SMPL_basicModel_m_lbs_path,
}


m = 0.005
N = 55

start_time = time.time()
print(
    "Started to write annotations from 3D human meshes at %s"
    % time.asctime(time.localtime(start_time))
)

for i, meshi in enumerate(files, 0):

    # meshpath
    meshpath = files[i]

    calvis.calvis_clear()

    calvis.mesh_path(meshpath)
    calvis.load_trimesh()
    calvis.fit_SMPL_model_to_mesh(SMPL_basicModel, gender=genders[i])

    calvis.segmentation(N=N)

    calvis.assemble_mesh_signatur(m=m)

    calvis.assemble_slice_statistics()

    cc = calvis.chest_circumference()
    ccslice_2D, to_3D = cc.to_planar()

    wc = calvis.waist_circumference()
    wcslice_2D, to_3D = wc.to_planar()

    pc = calvis.pelvis_circumference()
    pcslice_2D, to_3D = pc.to_planar()

    # Print info
    print("Chest circunference length is: %s" % ccslice_2D.length)
    print("Waist circunference length is: %s" % wcslice_2D.length)
    print("Pelvis circunference length is: %s" % pcslice_2D.length)

    annotation_file = (
        cmu_dataset_annotation_path
        + meshpath[cmu_dataset_meshes_path_length:-4]
        + "_anno.json"
    )

    with open(annotation_file, "r") as fp:
        data = json.load(fp)
        betas = np.array([beta for beta in data["betas"]])

    with open(annotation_file, "w") as fp:
        # Write the betas. Since "the betas" is a matrix, we have to
        # 'listifyit'.
        json.dump(
            {
                "betas": betas.tolist(),
                "human_dimensions": {
                    "chest_circumference": ccslice_2D.length,
                    "waist_circumference": wcslice_2D.length,
                    "pelvis_circumference": pcslice_2D.length,
                },
            },
            fp,
            sort_keys=True,
            indent=4,
            ensure_ascii=False,
        )
    print("Saved annotations file in %s" % annotation_file)
    print("%s percent finished" % (i / len(files) * 100))

finish_time = time.time()

print(
    "Started to write annotations from 3D human meshes at %s"
    % time.asctime(time.localtime(start_time))
)

print(
    "Finished to write annotations from 3D human meshes %s"
    % time.asctime(time.localtime(finish_time))
)
elapsed_time = finish_time - start_time
print("Total time needed was %s seconds" % elapsed_time)
