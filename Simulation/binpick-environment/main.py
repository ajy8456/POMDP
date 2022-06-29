import os
import math
import numpy as np

import pybullet as p


from envs.binpick_env import BinPickEnv
from envs.frankapanda import FrankaPanda
from utils.process_pointcloud import generatePointCloud, removeHiddenPoints, pcBase2World, pcWorld2Camera, visualizePointCloud


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

    # GPU acceleration (linux only)
    '''
    import pkgutil
    egl = pkgutil.get_loader('eglRenderer')
    if (egl):
        eglPluginId = p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
    '''

    # Init environments and agents
    binpick_env = BinPickEnv(customURDFPath)
    panda = FrankaPanda()



    stepCount = 0
    while True:
        # For smooth rendering
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
    
        # Render camera
        binpick_env.render()

        # Generate pointclouds from the URDFs
        # TODO: Optimize camera parameters
        obj0_pos, obj0_orn = p.getBasePositionAndOrientation(binpick_env.object0Uid)
        obj0_pos = np.array(obj0_pos)
        obj0_R = np.reshape(p.getMatrixFromQuaternion(obj0_orn), (3, 3))
        obj0_pcd = generatePointCloud(binpick_env.object0Uid)
        pcBase2World(obj0_pcd, obj0_pos, obj0_R)

        # TODO: Fix camera coordinate transfrom
        #camera_target_pos = np.array(binpick_env.camera_target_pos)
        #camera_euler_orn = np.array(binpick_env.camera_euler_orn)
        #camera_R = np.reshape(p.getMatrixFromQuaternion(p.getQuaternionFromEuler(camera_euler_orn)), (3, 3))
        #pcWorld2Camera(obj0_pcd, camera_target_pos, camera_R)
        visualizePointCloud(obj0_pcd)

        obj1_pos, obj1_orn = p.getBasePositionAndOrientation(binpick_env.object1Uid)    
        obj1_R = np.reshape(p.getMatrixFromQuaternion(obj1_orn), (3, 3))
        obj1_pcd = generatePointCloud(binpick_env.object1Uid)
        pcBase2World(obj1_pcd, obj1_pos, obj1_R)
        visualizePointCloud(obj1_pcd)

        # TODO: hidden point removal.


        # Reset
        if False:
            binpick_env.reset()
            panda.reset()
            stepCount=0

        # Update pybullet
        p.stepSimulation()
        stepCount+=1


    p.disconnect()