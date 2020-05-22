import sys
import os
import bpy
from mathutils import Matrix

import time

cmu_dataset_path = os.path.abspath("./../CALVIS/dataset/cmu/")
cmu_dataset_meshes_path = os.path.join(cmu_dataset_path, "human_body_meshes/")


male_directory = os.path.join(cmu_dataset_meshes_path, "male")
png_male_path = os.path.join(cmu_dataset_path, "synthetic_images/200x200/male")
female_directory = os.path.join(cmu_dataset_meshes_path, "female")
png_female_path = os.path.join(
    cmu_dataset_path, "synthetic_images/200x200/female"
)
countfiles = 0

# init scene
scene = bpy.data.scenes["Scene"]
# blender < v 2.80
# scene.render.engine = "BLENDER_RENDER"
# scene.render.engine = "CYCLES"
scene.render.engine = "BLENDER_EEVEE"
# set camera properties and initial position
cam_ob = bpy.data.objects["Camera"]
#scene.objects.active = cam_ob
cam_ob.matrix_world = Matrix(
    (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 0.08715556561946869, -0.9961947202682495, -3.0),
        (0.0, 0.9961947202682495, 0.08715556561946869, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
)

cam_ob.data.lens = 60
# Clipping is related to the depth of field (DoF). Rendering will start and end
# at this depth values.
cam_ob.data.clip_start = 0.1
cam_ob.data.clip_end = 100
# (default) cam_ob.data.sensor_fit = AUTO
cam_ob.data.sensor_width = 32
# (default) cam_ob.data.sensor_hight = 18
cam_ob.data.type = "ORTHO"
cam_ob.data.ortho_scale = 2.5
# blender < v 2.80
# cam_ob.data.draw_size = 0.5

# delete the default cube (which held the material) if any
try:
    bpy.data.objects["Cube"].select = True
    bpy.ops.object.delete(use_global=False)
except:
    print("No default cube found")

# position the lamp in front of the camara as simulating a "real lamp"
# hanging from the cilling
lamp_objects = [o for o in bpy.data.objects if o.type == "LAMP"]

# if there are no lamps we add one
if len(lamp_objects) == 0:
    # Create new lamp datablock
    # blender < v 2.80
    # lamp_data = bpy.data.lamps.new(name="Lamp", type="POINT")

    lamp_data = bpy.data.lights.new(name="Lamp", type='POINT')

    # Create new object with our lamp datablock
    lamp = bpy.data.objects.new(name="Lamp", object_data=lamp_data)

    # Link lamp object to the scene so it'll appear in this scene
    # blender < v 2.80
    # scene.objects.link(lamp)

    # link light object
    bpy.context.collection.objects.link(lamp)
else:
    lamp = lamp_objects[0]

lamp.matrix_world = Matrix(
    (
        (1.0, 0.0, 0.0, 0.10113668441772461),
        (0.0, 1.0, 0.0, -0.8406344056129456),
        (0.0, 0.0, 1.0, 1.4507088661193848),
        (0.0, 0.0, 0.0, 1.0),
    )
)

# 200 pixels in the x direction
bpy.context.scene.render.resolution_x = 200
# 200 pixels in the y direction
bpy.context.scene.render.resolution_y = 200
# "Percentage scale for render resolution"
bpy.context.scene.render.resolution_percentage = 100

# save a grayscale(BlackWhite) png image
scene.render.image_settings.color_mode = "BW"
scene.render.image_settings.file_format = "PNG"
# set the background color to white (R, G, B)
# blender < v2.80
# scene.world.horizon_color = (1, 1, 1)
scene.world.color = (1, 1, 1)
# scene.cycles.device = 'GPU'

bigbang_time = time.time()
for subdir, dirs, files in os.walk(female_directory):
    files.sort()
    start_time = time.time()
    print("Started to render pictures from female humans at %s" % start_time)
    for file in files:
        meshpath = os.path.join(subdir, file)
        savepng = os.path.join(png_female_path, file[:-3] + "png")
        # print(meshpath)
        # print(savepng)

        # Deselect everthing
        bpy.ops.object.select_all(action="DESELECT")

        imported_object = bpy.ops.import_scene.obj(filepath=meshpath)
        obj_object = bpy.context.selected_objects[0]
        # print('Imported name: ', obj_object.name)
        obj_object.data.use_auto_smooth = False  # autosmooth creates artifacts
        # assign the existing spherical harmonics material
        obj_object.active_material = bpy.data.materials["Material"]
        # update scene, if needed
        # blender < v2.80
        # scene.update()
        dg = bpy.context.evaluated_depsgraph_get()
        dg.update()

        # blender < v2.80
        # scene.render.use_antialiasing = True
        bpy.context.scene.display.render_aa = 'OFF'
        scene.render.filepath = savepng
        # disable render output
        logfile = "/dev/null"
        open(logfile, "a").close()
        old = os.dup(1)
        sys.stdout.flush()
        os.close(1)
        os.open(logfile, os.O_WRONLY)

        # Render
        bpy.ops.render.render(write_still=True)

        # now delete this mesh and update
        bpy.data.objects[obj_object.name].select = True
        bpy.ops.object.delete(use_global=False)
        # print('Object %s deleted.' % obj_object.name)
        to_remove = [block for block in bpy.data.meshes if block.users == 0]
        for block in to_remove:
            bpy.data.meshes.remove(block)

        # blender < v2.80
        # scene.update()
        dg = bpy.context.evaluated_depsgraph_get()
        dg.update()

        # disable output redirection
        os.close(1)
        os.dup(old)
        os.close(old)

for subdir, dirs, files in os.walk(male_directory):
    files.sort()
    start_time = time.time()
    print("Started to render pictures from male humans at %s" % start_time)
    for file in files:
        meshpath = os.path.join(subdir, file)
        savepng = os.path.join(png_male_path, file[:-3] + "png")
        # print(meshpath)
        # print(savepng)

        # Deselect everthing
        bpy.ops.object.select_all(action="DESELECT")

        imported_object = bpy.ops.import_scene.obj(filepath=meshpath)
        obj_object = bpy.context.selected_objects[0]
        # print('Imported name: ', obj_object.name)
        obj_object.data.use_auto_smooth = False  # autosmooth creates artifacts
        # assign the existing spherical harmonics material
        obj_object.active_material = bpy.data.materials["Material"]
        scene.update()
        # blender < v2.80
        # scene.render.use_antialiasing = False
        scene.render.render_aa = False
        scene.render.filepath = savepng
        # disable render output
        logfile = "/dev/null"
        open(logfile, "a").close()
        old = os.dup(1)
        sys.stdout.flush()
        os.close(1)
        os.open(logfile, os.O_WRONLY)

        # Render
        bpy.ops.render.render(write_still=True)

        # now delete this mesh and update
        bpy.data.objects[obj_object.name].select = True
        bpy.ops.object.delete(use_global=False)
        # print('Object %s deleted.' % obj_object.name)
        to_remove = [block for block in bpy.data.meshes if block.users == 0]
        for block in to_remove:
            bpy.data.meshes.remove(block)

        scene.update()

        # disable output redirection
        os.close(1)
        os.dup(old)
        os.close(old)

finish_time = time.time()
print(
    "Started to render pictures from humans at %s"
    % time.asctime(time.localtime(bigbang_time))
)
print(
    "Finished to render pictures from humans at %s"
    % time.asctime(time.localtime(finish_time))
)
elapsed_time = finish_time - bigbang_time
print("Total time needed was %s seconds" % elapsed_time)
