import os
import math
import numpy as np
import pickle

import pybullet as p

from envs.binpick_env import BinPickEnv
from envs.frankapanda import FrankaPanda

NUM_DATA = 5

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

    # PCD count
    pcd_count = 0

    # Simulation loop
    while True:
        # For smooth rendering
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
    
        # Render depth camera
        binpick_env.render()
        # Update environment step
        binpick_env.step()

        # Once pointcloud is generated, (None when the environment is not stabilized yet)
        pcd_gt = binpick_env.pcd_groundtruth
        if pcd_gt != None:
            # Collect data if pcd_gt is valid
            if len(pcd_gt) != 0:
                # Save as pickle
                print("Saving data...")
                with open("./dataset/pcd_groundtruth_"+str(pcd_count)+".bin", "wb") as file:
                    pickle.dump(pcd_gt, file)
                print("Data saved!" + str(pcd_count))
                pcd_count += 1
            else:
                print("Invalid environment")

            # Reset environment.
            binpick_env.reset()

            # Finish
            if pcd_count >= NUM_DATA: 
                break

        # Update pybullet
        p.stepSimulation()

    # Disconnet pybullet
    p.disconnect()
