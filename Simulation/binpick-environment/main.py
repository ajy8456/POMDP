import os
import math
import numpy as np

import pybullet as p


from envs.binpick_env import BinPickEnv
from envs.frankapanda import FrankaPanda
from utils.process_pointcloud import generatePointCloud, removeHiddenPoints, transformPointCloud, visualizePointCloud
from utils.process_geometry import matrixBase2World, matrixWorld2Camera


if __name__=="__main__":

    # Project path
    projectPath = os.path.dirname(os.path.abspath(__file__))
    customURDFPath = os.path.join(projectPath, "urdf")
    
    # Global simulatior setup
    CONTROL_DT = 1./240.
    p.connect(p.GUI)
    p.setTimeStep(CONTROL_DT)
    p.setGravity(0, 0, -9.8)
    p.resetDebugVisualizerCamera(cameraDistance=0.25, cameraYaw=240, cameraPitch=-40, cameraTargetPosition=[-0.25,0.20,0.8])

    # GPU acceleration (linux only) Reference: https://colab.research.google.com/drive/1u6j7JOqM05vUUjpVp5VNk0pd8q-vqGlx#scrollTo=fJXFN4U7NIRC
    import pkgutil
    egl = pkgutil.get_loader('eglRenderer')
    if (egl):
        eglPluginId = p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")

    # Init environments and agents
    binpick_env = BinPickEnv(customURDFPath)
    panda = FrankaPanda()

    # Simulation loop
    step_count = 0
    while True:
        # For smooth rendering
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
    
        # Render depth camera
        binpick_env.render()

        if step_count >= 50:
            # Generate pointclouds from the URDFs
            pcd = generatePointCloud(binpick_env.object0Uid)

            # Get transformation matrices
            T_world = matrixBase2World(binpick_env.object0Uid)
            T_camera = matrixWorld2Camera(binpick_env.view_matrix)
            T_base2camera = np.matmul(T_camera, T_world)
            
            # Transfrom the point cloud directly to the camera frame from the base frame.
            transformPointCloud(pcd, T_base2camera)
            visualizePointCloud(pcd)

            # TODO: hidden point removal.


        # Update pybullet
        p.stepSimulation()
        step_count+=1


    p.disconnect()