import os
import math

import pybullet as p
import pybullet_data

from utils.freeze_class import freeze_class


@freeze_class
class FrankaPanda:

    def __init__(self):

        # ==========
        # Kinematic config
        # ==========
        self.num_joints = 11
        self.num_joints_to_gripper = 9  
        self.current_pose = None
        self.rest_pose = (  
            0.0,        # Joint 0
            -0.6,       # Joint 1
            0.0,        # Joint 2
            -2.57,      # Joint 3
            0.0,        # Joint 4
            3.5,        # Joint 5
            0.75,       # Joint 6
            0.0,        # Joint 7
            0.0,        # Joint 8
            0.08,       # Joint 9 (right gripper)
            0.08        # Joint 10 (left gripper)
        )

        # ==========
        # Load URDF
        # ==========
        pybulletDataPath = pybullet_data.getDataPath()
        self.pandaUid = p.loadURDF(os.path.join(pybulletDataPath, "franka_panda/panda.urdf"),
                                                useFixedBase=True)
        
        # =========
        # Reset with initalization
        # =========
        self.reset()
    
    
    def reset(self):
        # Rest pose
        self.current_pose = list(self.rest_pose)
        # Feed pose value to the controller
        for i in range(self.num_joints_to_gripper):
            p.setJointMotorControl2(self.pandaUid, i, p.POSITION_CONTROL, self.current_pose[i])              # Arm
        for i in range(self.num_joints_to_gripper, self.num_joints):
            p.setJointMotorControl2(self.pandaUid, i, p.POSITION_CONTROL, self.current_pose[i], force=100)   # Gripper

        
    def step():
        pass
