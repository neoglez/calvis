import os
from vtkplotter import Plotter, trimesh2vtk, settings, write, Text2D
import locale
from calvis import Calvis

# change locale to load meshes in obj format with . as decimal separator.
locale.setlocale(locale.LC_NUMERIC, "C")


calvispath = "./../data/human_body_meshes/"
smpl_data_folder = os.path.abspath("./../datageneration/smpl_data")

SMPL_basicModel_f_lbs_path = os.path.join(
    smpl_data_folder, "basicModel_f_lbs_10_207_0_v1.0.0.pkl"
)
SMPL_basicModel_m_lbs_path = os.path.join(
    smpl_data_folder, "basicmodel_m_lbs_10_207_0_v1.0.0.pkl"
)

files = []
genders = []
meshes = []
# r=root, d=directories, f = files
for r, d, f in os.walk(calvispath):
    for file in f:
        if ".obj" in file:
            files.append(os.path.join(r, file))
            genders.append("female" if "female" == r[-6:] else "male")

SMPL_basicModel = {
    "f": SMPL_basicModel_f_lbs_path,
    "m": SMPL_basicModel_m_lbs_path,
}


# plotter
settings.embedWindow(backend=False)
vp = Plotter(shape=(2, 4), size=(500, 800), bg="w", axes=0)
#vp.sharecam = False
settings.useDepthPeeling = False
# Calvis
calvis = Calvis()

m = 0.005
N = 55

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
    cc_actor = trimesh2vtk(cc).unpack()[0].lw(2)

    wc = calvis.waist_circumference()
    wcslice_2D, to_3D = wc.to_planar()
    wc_actor = trimesh2vtk(wc).unpack()[0].lw(2)

    pc = calvis.pelvis_circumference()
    pcslice_2D, to_3D = pc.to_planar()
    pc_actor = trimesh2vtk(pc).unpack()[0].lw(2)

    text = Text2D("Subject no. %s" % (i + 1))

    # Print info
    #print("Chest circunference length is: %s" % ccslice_2D.length)
    #print("Waist circunference length is: %s" % wcslice_2D.length)
    #print("Pelvis circunference length is: %s" % pcslice_2D.length)

    # View the human dimensions
    slices = []
    slices.append(cc)
    slices.append(wc)
    slices.append(pc)

    human = vp.load(meshpath)

    vp.show(
        human.alpha(0.4), cc_actor, wc_actor, pc_actor, text, at=i
    )


vp.show(interactive=1)
