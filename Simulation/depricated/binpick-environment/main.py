import os
import math
import numpy as np

import pybullet as p

from envs.binpick_env import BinPickEnv
from envs.frankapanda import FrankaPanda


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
    while True:
        # For smooth rendering
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
    
        # Render depth camera
        binpick_env.render()








        # Update pybullet
        p.stepSimulation()

    p.disconnect()