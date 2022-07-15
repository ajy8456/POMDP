from logging import root
import os
import math
import numpy as np
import random
import matplotlib.pyplot as plt

import pybullet as p
import pybullet_data
from pybullet_object_models import ycb_objects

from utils.process_pointcloud import generatePointCloud, transformPointCloud, mergePointCloud, removeHiddenPoints, pointCloud2Dictionary, visualizePointCloud
from utils.process_geometry import matrixBase2World, matrixWorld2Camera, checkCollision

from utils.freeze_class import freeze_class

@freeze_class
class BinPickEnv():

    def __init__(self, customURDFPath):

        pybullet_data_path = pybullet_data.getDataPath()
        self.__step_count = 0
        self.__pcd_groundtruth = None
        self.__pcd_measurement = None        # Will be used later for camera meausurement.

        # ==========
        # Default objects config
        # ==========
        self.__plane_config = { 
            "urdf": os.path.join(pybullet_data_path, "plane.urdf"), 
            "pos": (0, 0, 0), 
            "orn": (0, 0, 0) 
        }
        self.__table_config = { 
            "urdf": os.path.join(customURDFPath, "cabinet.urdf"), 
            "pos": (0.7, 0, 0), 
            "orn": (0, 0, 0) 
        }
        self.__objects_config = (
            { 
                "urdf": os.path.join(ycb_objects.getDataPath(), "YcbHammer", "model.urdf"), 
                "pos": (0.7, 0.0, 0.57), 
                "orn": (0, 0, math.pi)      # Euler: roll, pitch, yaw
            },                
            { 
                "urdf": os.path.join(ycb_objects.getDataPath(), 'YcbBanana', "model.urdf"), 
                "pos": (0.7, 0.0, 0.57), 
                "orn": (0, 0, math.pi/2)    # Euler: roll, pitch, yaw
            }
        )

        # ==========
        # Random initialization config
        # ==========
        self.__jitter_pos = lambda : np.array([
            random.uniform(-0.08, 0.08), 
            random.uniform(-0.08, 0.08), 
            random.uniform(-0.03, 0.03)
        ])
        self.__jitter_orn = lambda : np.array([
            random.uniform(0, 0), 
            random.uniform(0, 0), 
            random.uniform(-math.pi, math.pi)
        ])

        # ==========
        # Load URDF id
        # ==========
        self.__plane_uid = p.loadURDF(self.__plane_config["urdf"], 
                                    basePosition = self.__plane_config["pos"], 
                                    baseOrientation = p.getQuaternionFromEuler(self.__plane_config["orn"]),
                                    useFixedBase=True)
        self.__table_uid = p.loadURDF(self.__table_config["urdf"],
                                    basePosition=self.__table_config["pos"],
                                    baseOrientation=p.getQuaternionFromEuler(self.__table_config["orn"]),
                                    useFixedBase=True)

        self.__objects_uid = []
        for obj_config in self.__objects_config:
            uid = p.loadURDF(obj_config["urdf"],
                            basePosition=obj_config["pos"],
                            baseOrientation=p.getQuaternionFromEuler(obj_config["orn"]),
                            useFixedBase=False)
            self.__objects_uid.append(uid)
        self.__objects_uid = tuple(self.__objects_uid)  # Make tuple to protect variable

        # ==========
        # RGB-D camera config
        # Reference: https://towardsdatascience.com/simulate-images-for-ml-in-pybullet-the-quick-easy-way-859035b2c9dd
        # ==========
        self.__camera_target_pos = [0.7, 0, 0.6]
        self.__camera_euler_orn = [0, 0, -90] # roll pitch yaw
        self.__view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.7, 0, 0.6],
                                                            distance=.35,
                                                            roll=0,
                                                            pitch=0,
                                                            yaw=-90,
                                                            upAxisIndex=2)
        self.__proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                                                    aspect=float(480)/float(360),
                                                    nearVal=0.1,
                                                    farVal=0.6)



    def step(self):

        # Generate pointcloud once stabilized
        if self.__step_count >= 50:
            # Reset count
            self.__step_count=0
            # Check collision
            objs_to_check_collision = [self.__table_uid, *self.__objects_uid]
            collision = checkCollision(objs_to_check_collision)

            if not collision:
                # Generating pointclouds of the target objects
                pcd_list = []
                for i in range(len(self.__objects_uid)):
                    # Generate pointcloud from the URDFs
                    _pcd = generatePointCloud(self.__objects_uid[i])
                    # Get transformation matrices
                    T_world = matrixBase2World(self.__objects_uid[i])
                    T_camera = matrixWorld2Camera(self.__view_matrix)
                    T_base2camera = np.matmul(T_camera, T_world)
                    # Transfrom the point cloud directly to the camera frame from the base frame.
                    transformPointCloud(_pcd, T_base2camera)
                    pcd_list.append(_pcd)

                # Merge point clouds    
                pcd_merged = mergePointCloud(pcd_list)
                # Hidden point removal.
                pcd_occluded, pt_map = removeHiddenPoints(pcd_merged, [0, 0, 0])
                # Create point cloud dictionary
                pcd_dict = pointCloud2Dictionary(pcd_merged, pt_map)
                # Visualize
                # visualizePointCloud(pcd_dict)
                # Save ground truth point cloud
                self.__pcd_groundtruth = pcd_dict
            else:
                # Create empty dictionary if collision happens
                self.__pcd_groundtruth = {}

        # Update step count
        self.__step_count += 1



    def reinitialize_object(self, i):
        '''
        Params
        - i: index of the objects to reinitalize
        '''
        pos = np.array(self.__objects_config[i]["pos"]) + self.__jitter_pos()
        orn = np.array(self.__objects_config[i]["orn"]) + self.__jitter_orn()
        p.resetBasePositionAndOrientation(self.__objects_uid[i], pos, p.getQuaternionFromEuler(orn))



    def reset(self):
        '''
        Reset environment.
        1. Erase __pcd_groundtruth, __pcd_measurement
        2. Reinitialize objects
        '''
        self.__pcd_groundtruth = None
        self.__pcd_measurement = None

        for i in range(len(self.__objects_uid)):
            self.reinitialize_object(i)



    def render(self):
        '''
        Setup the RGB-D camera in pybullet.

        Process includes...
        1. (Pre)Get __view_matrix
        2. (Pre)Get __proj_matrix from __view_matrix
        3. (Runtime)Get Image from __proj_matrix
        4. (Runtime)Convert values to np.array

        References
        - https://towardsdatascience.com/simulate-images-for-ml-in-pybullet-the-quick-easy-way-859035b2c9dd
        '''
        # Get camera image from simulation
        (w, h, px, px_d, px_id) = p.getCameraImage(width=480,
                                            height=360,
                                            viewMatrix=self.__view_matrix,
                                            projectionMatrix=self.__proj_matrix,
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


    @property
    def pcd_groundtruth(self):
        return self.__pcd_groundtruth

    
    @property
    def pcd_measurement(self):
        return self.__pcd_measurement


