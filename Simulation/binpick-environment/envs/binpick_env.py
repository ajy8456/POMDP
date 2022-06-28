import os
import math
import numpy as np
import matplotlib.pyplot as plt

import pybullet as p
import pybullet_data
from pybullet_object_models import ycb_objects

from utils.freeze_class import freeze_class

@freeze_class
class BinPickEnv():

    def __init__(self, customURDFPath):
        # ==========
        # Objects config
        # ==========
        self.plane_pos = (0, 0, 0)
        self.table_pos = (0.7, 0, 0)
        self.object0_pos = (0.7, 0.0, 0.53)
        self.object0_orn = (0, 0, math.pi)  # Euler: roll, pitch, yaw
        self.object1_pos = (0.58, 0.0, 0.53) 
        self.object1_orn = (0, 0, math.pi/2)                        # Euler: roll, pitch, yaw

        # ==========
        # Load URDF
        # ==========
        pybulletDataPath = pybullet_data.getDataPath()
        self.planeUid = p.loadURDF(os.path.join(pybulletDataPath, "plane.urdf"), 
                                    basePosition=self.plane_pos)
        self.tableUid = p.loadURDF(os.path.join(customURDFPath, "cabinet.urdf"), 
                                    basePosition=self.table_pos, useFixedBase=True)
        self.object0Uid = p.loadURDF(os.path.join(ycb_objects.getDataPath(), 'YcbHammer', "model.urdf"), 
                                    basePosition=self.object0_pos, baseOrientation=p.getQuaternionFromEuler(self.object0_orn))
        self.object1Uid = p.loadURDF(os.path.join(ycb_objects.getDataPath(), 'YcbBanana', "model.urdf"),
                                    basePosition=self.object1_pos, baseOrientation=p.getQuaternionFromEuler(self.object1_orn))

        # ==========
        # RGB-D camera config
        # Reference: https://towardsdatascience.com/simulate-images-for-ml-in-pybullet-the-quick-easy-way-859035b2c9dd
        # ==========
        self.camera_target_pos = [0.7, 0, 0.6]
        self.camera_euler_orn = [0, 0, -90] # roll pitch yaw
        self.view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.7, 0, 0.6],
                                                            distance=.35,
                                                            roll=0,
                                                            pitch=0,
                                                            yaw=-90,
                                                            upAxisIndex=2)

        self.proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                                                    aspect=float(480)/float(360),
                                                    nearVal=0.1,
                                                    farVal=0.4)



    def reset(self):
        pass

    def step(self):
        pass

    def render(self):
        '''
        Setup the RGB-D camera in pybullet.

        Process includes...
        1. (Pre)Get view_matrix
        2. (Pre)Get proj_matrix from view_matrix
        3. (Runtime)Get Image from proj_matrix
        4. (Runtime)Convert values to np.array

        References
        - https://towardsdatascience.com/simulate-images-for-ml-in-pybullet-the-quick-easy-way-859035b2c9dd
        '''
        # Get camera image from simulation
        (w, h, px, px_d, px_id) = p.getCameraImage(width=480,
                                            height=360,
                                            viewMatrix=self.view_matrix,
                                            projectionMatrix=self.proj_matrix,
                                            renderer=p.ER_BULLET_HARDWARE_OPENGL)

        # Reshape list into ndarray(image)
        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (w, h, 4))
        rgb_array = rgb_array[:, :, :3]                 # remove alpha

        depth_array = np.array(px_d, dtype=np.float32)
        depth_array = np.reshape(depth_array, (w, h))

        mask_array = np.array(px_id, dtype=np.uint8)
        mask_array = np.reshape(mask_array, (w, h))
        
        # Return images
        return rgb_array, depth_array, mask_array
