import pybullet as p
import pybullet_data
import math
import numpy as np
import random
from pybullet_tools.utils import plan_joint_motion, set_joint_positions, wait_for_duration

INF = 1e+9

class PandaEnv():
    def __init__(self):     # connect to GUI(to visualize) and set the camera view(humans view)
        self.step_counter = 0
        p.connect(p.GUI)
        p.resetDebugVisualizerCamera(cameraDistance=1.7, cameraYaw=180., cameraPitch=-50, cameraTargetPosition=[.75,.2,0.2])

    def reset(self, obj1_pos=(0,0,0), obj1_ori=0, obj2_pos=(0,0,0), obj2_ori=0):       # Initialize the environment
        objcount = 2
        objwidth = 0.1

        if obj1_pos==obj2_pos==(0, 0, 0):
            objpos = [(0.7*i + random.uniform(0.1, 0.5), random.uniform(0.4, 0.6), 0.08) for i in range(objcount)]
            objori = [(0, 0, random.uniform(0, math.pi/8)) for _ in range(objcount)]
        else :
            obj1_ori = (obj1_ori + math.pi)%(math.pi/8)
            obj2_ori = (obj2_ori + math.pi)%(math.pi/8)
            '''
            obj_pos : the position of the object's left-most corner
            center of the object1 == obj1_pos + tup1
            '''
            tup1 = (((1/math.sqrt(2))*(objwidth/2)*math.cos(math.pi/4+obj1_ori)), ((1 / math.sqrt(2)) * (objwidth / 2) * math.sin(math.pi / 4 + obj1_ori)), 0)
            tup2 = (((1/math.sqrt(2))*(objwidth/2)*math.cos(math.pi/4+obj2_ori)), ((1 / math.sqrt(2)) * (objwidth / 2) * math.sin(math.pi / 4 + obj2_ori)), 0)

            obj1_pos=tuple(sum(elem) for elem in zip(obj1_pos, tup1))
            obj2_pos=tuple(sum(elem) for elem in zip(obj2_pos, tup2))
            objpos = [obj1_pos, obj2_pos]
            objori = [(0, 0, obj1_ori), (0, 0, obj2_ori)]

        p.resetSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, True)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, True)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0,0,-9.8)

        self.pandaUid = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True , basePosition=[0.75, -.3, -0.1], baseOrientation=p.getQuaternionFromEuler([0, 0, math.pi/2]))
        self.tableUid = p.loadURDF("table/table.urdf", basePosition=[0.75, 0.5, -0.65], globalScaling = 1.001)
        self.plainID = p.loadURDF("plane.urdf", basePosition=[0, 0, -0.65])
        self.objectUid = []
        for i in range(objcount) :
            self.objectUid.append(p.loadURDF("simulation/longcube.urdf", basePosition = objpos[i], baseOrientation=p.getQuaternionFromEuler(objori[i]), globalScaling = 0.1))

        rest_poses = [0, -0.215, 0, -2.57, 0, 2.356, 2.356, 0.08, 0.08]
        for i in range(7):
            p.resetJointState(self.pandaUid, i, rest_poses[i])
        p.resetJointState(self.pandaUid, 9, 0.08)
        p.resetJointState(self.pandaUid, 10, 0.08)
        # self.state_robot = p.getLinkState(self.pandaUid, 11)[0]  # the position of end-effector
        # self.orien_robot = p.getLinkState(self.pandaUid, 11)[1]
        # self.state_fingers = (p.getJointState(self.pandaUid, 9)[0],
        #                  p.getJointState(self.pandaUid, 10)[0])  # each finger's distance to the center
        #
        # self.panda_position = self.state_robot + self.state_fingers
        return np.array(self.get_panda_position()).astype(np.float32)

    def step(self, goal_pos, goal_orien, fingers):

        ll = [-2.9671, -1.8326, -2.9671, -3.1416, -2.9671, -0.0873, -2.9671]
        ul = [2.9671, 1.8326, 2.9671, 0.0, 2.9671, 3.8223, 2.9671]
        jr = [5.8, 3.6, 5.8, 2.9, 5.8, 3.8, 5.8]
        rp = [0, -0.215, 0, -2.57, 0, 2.356, 2.356]
        jointPoses = p.calculateInverseKinematics(self.pandaUid,11,goal_pos,goal_orien, ll, ul, jr, rp)[0:7]
        # print('joint poses : ', jointPoses)

        obstacles = [
            self.tableUid,
            self.objectUid[0],
            self.objectUid[1]
        ]
        path_conf = plan_joint_motion(self.pandaUid, list(range(7)), jointPoses,
                                      obstacles=obstacles, algorithm='birrt')
        
        # self.render()
        
        # path_conf = plan_joint_motion(self.pandaUid, list(range(7)), jointPoses, max_distance=1e-5,
                                      # obstacles=obstacles)
        # if path_conf is None:
        #     print('Unable to find a path')
        #     return
        if path_conf is not None :
            for q in path_conf:
                self.render()
                # set_joint_positions(self.pandaUid, list(range(7))+[9,10], list(q)+[0.04,0.04])
                p.setJointMotorControlArray(self.pandaUid, list(range(7))+[9,10], p.POSITION_CONTROL, list(q)+2*[fingers])
                p.stepSimulation()

        print('this is the distance : ', self.observation())
        reward = 0  # not use reward var
        done = False
        return np.array(self.get_panda_position()).astype(np.float32), reward, done, self.observation()

    def get_state_robot(self):
        return p.getLinkState(self.pandaUid, 11)[0]

    def get_orien_robot(self):
        return p.getLinkState(self.pandaUid, 11)[1]

    def get_state_fingers(self):
        return (p.getJointState(self.pandaUid, 9)[0], p.getJointState(self.pandaUid, 10)[0])

    def get_panda_position(self):
        return self.get_state_robot() + self.get_state_fingers()

    def observation(self, width = 720, height = 720, far = 100.0, near = 0.01):
        depth = far * near / (far - (far - near) * self.dp) # calculate the distance
        distance = depth[width // 2, height // 2]
        if distance > far*0.99 : return INF
        return distance

    def get_pos_robot(self):
        return self.get_state_robot() + self.get_orien_robot()

    def get_link_position(self):
        return p.getLinkState(self.pandaUid, 9)[0]

    def get_link_orien(self):
        return p.getLinkState(self.pandaUid, 9)[1]

    def debugparameter(self, start):
        motorsIds = []
        pos = self.get_pos_robot()
        # print('obser : ',obser)

        xyz = pos[:3]
        ori = p.getEulerFromQuaternion(pos[3:])
        # ori = obser[4:]
        xin = xyz[0]
        yin = xyz[1]
        zin = xyz[2]
        rin = ori[0]
        pitchin = ori[1]
        yawin = ori[2]
        abs_distance = 1.0
        p.addUserDebugParameter("Camera X", -3, 3, 0.75)
        p.addUserDebugParameter("Camera Y", -3, 3, 0.2)
        p.addUserDebugParameter("Camera Z", -3, 3, 0.2)
        motorsIds.append(p.addUserDebugParameter("X", 0, 1.5, xin))
        motorsIds.append(p.addUserDebugParameter("Y", 0, abs_distance, yin))
        motorsIds.append(p.addUserDebugParameter("Z", 0, abs_distance, zin))
        motorsIds.append(p.addUserDebugParameter("roll", -math.pi, math.pi, rin))
        motorsIds.append(p.addUserDebugParameter("pitch", -math.pi, math.pi, pitchin))
        motorsIds.append(p.addUserDebugParameter("yaw", -math.pi, math.pi, yawin))
        motorsIds.append(p.addUserDebugParameter("fingerAngle", 0, 0.1, .04))

        while(True) :
            action = []
            for motorId in motorsIds :
                action.append(p.readUserDebugParameter(motorId))
            goal_pos = action[0:3]
            goal_orien = p.getQuaternionFromEuler(action[3:6])
            fingers = action[6]

            camera = p.getDebugVisualizerCamera()
            p.resetDebugVisualizerCamera(camera[-2], camera[-4], camera[-3], [p.readUserDebugParameter(0),p.readUserDebugParameter(1),p.readUserDebugParameter(2)])
            panda_position, reward, done, observation = self.step(goal_pos, goal_orien, fingers)
            # print(self.simulator_get_observation(robot_pos=self.get_link_position(), robot_orn=self.get_link_orien()))
            # print('*',env.step(start, goal_pos, goal_orien, fingers))
            # start = panda_position


    def render(self, width = 720, height = 720, far = 100.0, near = 0.01):       # Just camera rendering function

        dis = 1000.0
        # self.state_robot = p.getLinkState(self.pandaUid, 11)[0]  # the position of end-effector
        # self.orien_robot = p.getLinkState(self.pandaUid, 11)[1]

        offset_pos = [0, 0, 0.035]
        target_vec = [-1, 0, 0]
        offset_orn = p.getQuaternionFromEuler([0, 0, 0])

        view_pos, view_orn = p.multiplyTransforms(self.get_link_position(), self.get_link_orien(), offset_pos, offset_orn)
        target_pos, _ = p.multiplyTransforms(view_pos, view_orn, offset_pos, offset_orn)
        target_vector, _ = p.multiplyTransforms(view_pos, view_orn, target_vec, offset_orn)

        view_matrix = p.computeViewMatrix(cameraEyePosition= view_pos,
                                          cameraTargetPosition = target_pos,
                                          cameraUpVector = target_vector
                                          )
        proj_matrix = p.computeProjectionMatrixFOV(fov=80,
                                                     aspect=float(width) /height,
                                                     nearVal=near,
                                                     farVal=far)
        (_, _, px, self.dp, _) = p.getCameraImage(width=width,
                                              height=height,
                                              viewMatrix=view_matrix,
                                              projectionMatrix=proj_matrix,
                                              renderer=p.ER_BULLET_HARDWARE_OPENGL)

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (width,height, 4))

        rgb_array = rgb_array[:, :, :3]
        return rgb_array, self.dp

    def simulator_get_observation(self, robot_pos=False, robot_orn=False, width=720, height=720, far=100.0, near=0.01):
        if robot_pos==False or robot_orn==False :
            return -1
        offset_pos = [0, 0, 0.035]
        offset_orn = p.getQuaternionFromEuler([0, 0, 0])
        view_pos, view_orn = p.multiplyTransforms(robot_pos,robot_orn, offset_pos, offset_orn)
        target_pos, _ = p.multiplyTransforms(view_pos, view_orn, [0, 0, 5], offset_orn)

        # print(p.rayTest(view_pos, target_pos))
        objectid, _, _, hitposition, _ = p.rayTest(view_pos, target_pos)[0]
        if objectid == -1 :
            return INF
        else :
            return math.dist(view_pos, hitposition)

    def close(self):
        p.disconnect()
