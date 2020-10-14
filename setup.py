"""
Created on Wed Oct 14 15:40:18 2020

@author: neoglez
"""
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
    
REQUIREMENTS = ["numpy", "opencv-python", "trimesh"]

setuptools.setup(
     name='calvis',  
     version='0.0.1',
     #scripts=['dokr'],
     author="Yansel Gonzalez Tejeda",
     author_email="neoglez@gmail.com",
     install_requires=REQUIREMENTS,
     description="Chest, wAist and pelVIS circumference from 3D human Body meshes for Deep Learning",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/neoglez/calvis",
     packages=setuptools.find_packages(
         exclude=[
             "CALVIS",
             "code",
             "data",
             "datageneration",
             "img",
             "notebook"]),
     classifiers=[
         "Programming Language :: Python :: 2",
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
     

 )
