#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 10:55:01 2019

@author: neoglez
"""
import trimesh
import trimesh.viewer
import numpy as np
import pandas as pd
from vtkplotter import trimesh2vtk
import os
import pickle
import hbm
import locale

locale.setlocale(locale.LC_NUMERIC, "C")


class Calvis:
    """
    A class to calculate chest, waist and 
    pelvis circumference from 3D human body meshes.
    """

    def __init__(self):
        # mesh path on file system
        self.meshpath = None
        #
        self.mesh = None
        # mesh loaded with trimesh library
        self.trimesh = None
        # O(bject) SMPL female model
        self.smpl_female_model = None
        # O(bject) SMPL male model
        self.smpl_male_model = None
        # O(bject) SMPL human model initialized for the concrete mesh
        self.human_model = None
        # Joint locations for this concrete human model
        self.joints_location = None
        # Sliding vector: elements of this vector are points along the y
        # axis where the sliding plane intersects the mesh.
        self.sliding_vector = np.array([])
        # The Point $p_c$ that we consider the axilla center.
        self.axilla_center = None
        # The Point $p_a$ that we consider the axilla lower bound and
        # threfore the chest region upper bound
        self.axilla_lower_bound = None
        # Array of points along the y axis that define the region baoundaries.
        self.regions_bound = None
        # Vector which elements are the boundary length of the intersections
        # defined by the plane normal to the floor, the mesh and the points in
        # the sliding vector.
        self.mesh_signature = np.array([])
        # Slice additional information. Used to conduct queries.
        self.slice_statistics = np.array([])
        # Ray iota with origin in the right shoulder joint in direction to the
        # middle left inferior edge of the bounding box.
        self.iota = np.array([])

    def calvis_clear(self):
        """
        Reset all state variables to its default value. This method is a
        rudimentary way of making house cleaning.
        """
        self.__init__()

    def mesh_path(self, meshpath):
        self.meshpath = meshpath
        return self

    def load_trimesh(self):
        """
        Load the mesh with trimesh library
        """
        self.trimesh = trimesh.load_mesh(self.meshpath)
        return self

    def global_maximum(self):
        return (self.mg_index, self.mg)

    def make_slice(self, y):
        """
        Returns a slice "parallel" to the floor at point with given y 
        coordinate.
        """
        return self.trimesh.section(
            plane_origin=np.array([0, y, 0]), plane_normal=[0, 1, 0]
        )

    def chest_circumference(self):
        """
        Returns a slice representing the chest circumference
        """

        # Maximum boundary length in chest region
        bounds = self.regions_bound
        chest_upper_bound = bounds[0]
        chest_lower_bound_waist_upper_bound = bounds[1]

        chest_region = self.slice_statistics.query(
            "ycoordinate <= %f and ycoordinate > %f"
            % (chest_upper_bound, chest_lower_bound_waist_upper_bound)
        )
        chest_circumference_index = chest_region["evalue"].idxmax()
        chest_circumference_y = self.sliding_vector[chest_circumference_index]

        return self.trimesh.section(
            plane_origin=np.array([0, chest_circumference_y, 0]),
            plane_normal=[0, 1, 0],
        )

    def waist_circumference(self):
        """
        Returns a slice representing the waist circumference
        """
        # Maximum boundary length in chest region
        bounds = self.regions_bound
        waist_upper_bound = bounds[1]
        waist_lower_bound_pelvis_upper_bound = bounds[2]

        waist_region = self.slice_statistics.query(
            "ycoordinate <= %f and ycoordinate > %f"
            % (waist_upper_bound, waist_lower_bound_pelvis_upper_bound)
        )
        waist_circumferenc_index = waist_region["evalue"].idxmin()
        waist_circumferenc_y = self.sliding_vector[waist_circumferenc_index]

        return self.trimesh.section(
            plane_origin=np.array([0, waist_circumferenc_y, 0]),
            plane_normal=[0, 1, 0],
        )

    def pelvis_circumference(self):
        """
        Returns a slice representing the pelvis circumference
        """
        # Maximum boundary length in chest region
        bounds = self.regions_bound
        waist_lower_bound_pelvis_upper_bound = bounds[2]
        pelvis_lower_bound = bounds[3]

        pelvis_region = self.slice_statistics.query(
            "ycoordinate <= %f and ycoordinate > %f"
            % (waist_lower_bound_pelvis_upper_bound, pelvis_lower_bound)
        )
        pelvis_circumferenc_index = pelvis_region["evalue"].idxmax()
        pelvis_circumferenc_y = self.sliding_vector[pelvis_circumferenc_index]

        return self.trimesh.section(
            plane_origin=np.array([0, pelvis_circumferenc_y, 0]),
            plane_normal=[0, 1, 0],
        )

    def assemble_mesh_signatur(self, m=0.001):
        """
        Assembles the mesh signature.

        Parameters
        -----------
        m: float
            Slice the mesh every m-meters.
        """

        # we're going to slice the mesh into evenly spaced chunks along y
        # this takes the (2,3) bounding box and slices it into [miny, maxy]
        y_extents = self.trimesh.bounds[:, 1]
        # slice every m model units (eg, meters, dc, etc.) = every 1 mm.
        self.sliding_vector = np.flip(np.arange(*y_extents, step=m))

        intersections_boundary_length = []

        for idx, j in enumerate(self.sliding_vector):
            mslice = self.trimesh.section(
                plane_origin=np.array([0, self.sliding_vector[idx], 0]),
                plane_normal=[0, 1, 0],
            )

            # print(type(mslice))
            if mslice is None:
                intersections_boundary_length.append(0)
            else:
                slice_2D, to_3D = mslice.to_planar()
                intersections_boundary_length.append(slice_2D.length)

        self.mesh_signature = np.array(
            [elem for elem in intersections_boundary_length]
        )
        return self

    def segmentation(self, N=10):
        """
        Returns the chest, waist and pelvis regions bounds. The lower bound of
        one region is upper bound of the nextone. Therefore, the function
        returns (chest_upper_bound, chest_lower_bound/waist_upper_bound,
        waist_lower_bound/pelvis_upper_bound, pelvis_lower_bound).
    
        Parameters
        -----------
        N: integer
            Number of nearest neighbours relative to the axilla center
            to search for.
        """
        chest_upper_bound = None
        chest_lower_bound_waist_upper_bound = None
        waist_lower_bound_pelvis_upper_bound = None
        pelvis_lower_bound = None

        pa = self.recognize_axilla(N=N)
        joints_location = self.joints_location
        inverted_joint_names = dict(
            (v, k)
            for k, v in self.human_model.mean_template_shape.joint_names.items()
        )

        natural_waist_xyz = joints_location[inverted_joint_names["Spine1"]]
        pelvis_xyz = joints_location[inverted_joint_names["Pelvis"]]
        hip_xyz = joints_location[inverted_joint_names["R_Hip"]]

        chest_upper_bound = pa[0][1]
        chest_lower_bound_waist_upper_bound = natural_waist_xyz[1]
        waist_lower_bound_pelvis_upper_bound = pelvis_xyz[1]
        pelvis_lower_bound = hip_xyz[1]

        self.regions_bound = np.array(
            [
                chest_upper_bound,
                chest_lower_bound_waist_upper_bound,
                waist_lower_bound_pelvis_upper_bound,
                pelvis_lower_bound,
            ]
        )
        return self.regions_bound

    def recognize_axilla(self, N=10):
        """
        Returns the axilla point $p_a$
        
        Parameters
        -----------
        N: integer
            Number of nearest neighbours relative to the axilla center
            to search for.
        
        """
        # cast a ray with origin in the shoulder and direction d.
        ray_params = self.ray_iota()
        ray_origins = ray_params[0]
        ray_directions = ray_params[1]

        # run the mesh- ray query
        locations, index_ray, index_tri = self.trimesh.ray.intersects_location(
            ray_origins=ray_origins, ray_directions=ray_directions
        )
        self.axilla_center = locations[0]

        # find the N nearest neighbors.
        # We have to this with vtkplotter because trimesh does not have this
        # functionality implemented (jet)
        vtkactor = trimesh2vtk(self.trimesh)

        nearest_neighbours = vtkactor.closestPoint(self.axilla_center, N=N)
        sorted_Y = nearest_neighbours[nearest_neighbours[:, 1].argsort()]
        self.axilla_lower_bound = sorted_Y[:1]
        return self.axilla_lower_bound

    def assemble_slice_statistics(self):
        global_max = np.argmax(self.mesh_signature)
        global_min = np.argmin(self.mesh_signature)
        stats_list = []

        for i, elem in enumerate(self.mesh_signature, 0):
            is_global_max = i == global_max
            is_global_min = i == global_min
            try:
                is_local_max = (self.mesh_signature[i - 1] < elem) and (
                    elem > self.mesh_signature[i + 1]
                )
            except:
                # uncomment to debug
                # print(i)
                pass

            try:
                is_local_min = (self.mesh_signature[i - 1] > elem) and (
                    elem < self.mesh_signature[i + 1]
                )
            except:
                # uncomment to debug
                # print(i)
                pass

            # construct a node for every element
            node = {
                "eindex": i,  # slice index
                "evalue": elem,  # boundary length (real number)
                "ycoordinate": self.sliding_vector[i],  # y-coordinate
                "localmax": False or is_local_max,
                "localmin": False or is_local_min,
                "globalmax": False or is_global_max,  # global maximum?
                "globalmin": False or is_global_min,  # global minimum?
            }
            stats_list.append(node)

        # Convert to pandas to conduct queries
        self.slice_statistics = pd.DataFrame(stats_list)

    def fit_SMPL_model_to_mesh(self, smplmodel, gender):

        SMPL_basicModel_f_lbs_path = smplmodel["f"]
        SMPL_basicModel_m_lbs_path = smplmodel["m"]

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
        self.human_female_model = hbm.OSmplWithPose(
            new_female_template, female_model.get("shapedirs").x, None, None
        )
        self.human_male_model = hbm.OSmplWithPose(
            new_male_template, male_model.get("shapedirs").x, None, None
        )

        # select gender
        self.human_model = (
            self.human_male_model
            if (gender == "male" or gender == "m")
            else self.human_female_model
        )

        self.human_model.read_from_obj_file(self.meshpath)

        # recompute skeleton for this skinned body
        if self.human_model.must_recompute_skeleton:
            self.joints_location = self.human_model.recompute_skeleton()
        else:
            self.joints_location = (
                self.human_model.mean_template_shape.joints_location
            )
        return self

    def calculate_stature_from_mesh(self, mesh):
        # Since the mesh has LSA orientation, we continue as follows:
        vertices = self.calvis.trimesh.vertices
        sorted_Y = self.calvis.trimesh.vertices[vertices[:, 1].argsort()]
        #################################################
        # Take the two vertices to compute the distance #
        #################################################
        vertex_with_smallest_y = sorted_Y[:1]
        vertex_with_biggest_y = sorted_Y[6889:]
        # Simulate the 'floor' by setting the x and z coordinate to 0.
        vertex_on_floor = np.array([0, vertex_with_smallest_y[0, 1], 0])
        stature = np.linalg.norm(
            np.subtract(vertex_with_biggest_y, vertex_on_floor)
        )
        return stature, vertex_with_biggest_y, vertex_on_floor

    def chest_region(self):
        chest_upper_bound = self.regions_bound[0]
        chest_lower_bound_waist_upper_bound = self.regions_bound[1]
        return (chest_upper_bound, chest_lower_bound_waist_upper_bound)

    def waist_region(self):
        chest_lower_bound_waist_upper_bound = self.regions_bound[1]
        waist_lower_bound_pelvis_upper_bound = self.regions_bound[2]
        return (
            chest_lower_bound_waist_upper_bound,
            waist_lower_bound_pelvis_upper_bound,
        )

    def pelvis_region(self):
        waist_lower_bound_pelvis_upper_bound = self.regions_bound[2]
        pelvis_lower_bound = self.regions_bound[3]
        return (waist_lower_bound_pelvis_upper_bound, pelvis_lower_bound)

    def right_shoulder_joint(self):
        inverted_joint_names = dict(
            (v, k)
            for k, v in self.human_model.mean_template_shape.joint_names.items()
        )
        return self.joints_location[inverted_joint_names["R_Shoulder"]]

    def ray_iota(self):
        if len(self.iota) == 0:
            shoulder_xyz = self.right_shoulder_joint()
            # cast a ray with origin in the shoulder and direction d
            ray_origins = np.array([shoulder_xyz])
            ray_direction_outside_mesh = np.array(
                [
                    self.trimesh.bounding_box.bounds[0][0] - shoulder_xyz[0],
                    self.trimesh.bounding_box.bounds[0][1] - shoulder_xyz[1],
                    self.trimesh.bounding_box.bounds[1][2]
                    + self.trimesh.bounding_box.bounds[0][2]
                    + shoulder_xyz[2],
                ]
            )
            self.iota = np.array(
                [ray_origins, np.array([ray_direction_outside_mesh])]
            )

        return self.iota


if __name__ == "__main__":

    calvispath = "./../data/human_body_meshes"

    files = []
    genders = []
    meshes = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(calvispath):
        for file in f:
            if ".obj" in file:
                files.append(os.path.join(r, file))
                genders.append("female" if "female" == r[-6:] else "male")

    # mesh and annotation files
    male_subject_id = 2
    female_subject_id = 6

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

    # meshpath
    meshpath = files[female_subject_id]

    calvis = Calvis()
    calvis.mesh_path(meshpath)
    calvis.load_trimesh()
    calvis.fit_SMPL_model_to_mesh(
        SMPL_basicModel, gender=genders[female_subject_id]
    )
    regions_bounds = calvis.segmentation()

    chest_upper_bound = regions_bounds[0]
    chest_lower_bound_waist_upper_bound = regions_bounds[1]
    waist_lower_bound_pelvis_upper_bound = regions_bounds[2]
    pelvis_lower_bound = regions_bounds[3]

    calvis.assemble_mesh_signatur()
    ms = calvis.mesh_signature

    calvis.assemble_slice_statistics()

    # mg_index, mg = calvis.global_maximum()

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

    # View the human dimensions
    slices = []
    for y in regions_bounds:
        slices.append(calvis.make_slice(y))

    slices.append(cc)
    slices.append(wc)
    slices.append(pc)

    inverted_joint_names = dict(
        (v, k)
        for k, v in calvis.human_model.mean_template_shape.joint_names.items()
    )

    shoulder_xyz = calvis.joints_location[inverted_joint_names["R_Shoulder"]]
    ray_origins = np.array([shoulder_xyz])
    ray_direction_outside_mesh = [
        shoulder_xyz[0]
        + calvis.trimesh.bounding_box.bounds[0][0],  # shoulder_xyz[1] +
        calvis.trimesh.bounding_box.bounds[0][1],  # shoulder_xyz[2] +
        (
            (  # calvis.trimesh.bounding_box.bounds[1][2]
                calvis.trimesh.bounding_box.bounds[0][2]
            )
        ),
    ]

    ray_directions = np.array([ray_direction_outside_mesh])

    ray_visualize = trimesh.load_path(
        np.hstack(
            ([calvis.axilla_center], ray_origins + ray_directions)
        ).reshape(-1, 2, 3)
    )

    # axis
    geom = trimesh.creation.axis(0.02)

    scene = trimesh.Scene(
        [
            geom,
            calvis.trimesh,
            ray_visualize,
            # calvis.trimesh.bounding_box,
            # calvis.trimesh.bounding_box_oriented,
            # *slices,
            cc,
            wc,
            pc,
        ]
    )
    flags = {"wireframe": False}
    viewer = trimesh.viewer.SceneViewer(scene=scene, flags=flags)
