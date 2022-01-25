import pybullet as p
import pybullet_data
import math
import numpy as np


class PandaEnv(): # object name is 'duck' (not 'lego') because at first the object was three ducks(I changed because of some errors)
    def __init__(self):     # connect to GUI(to visualize) and set the camera view(humans view)
        self.step_counter = 0
        p.connect(p.GUI)
        p.resetDebugVisualizerCamera(cameraDistance=0.7, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0.55,-0.35,0.2])

    def duck_position(self, i):     #returns current object's position (lego1~3)
        self.duck = i - 1
        duckpos, _ = p.getBasePositionAndOrientation(self.objectUid[self.duck])
        return duckpos

    def step(self, action, process, duck):
        self.duck = duck
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
        orientation = p.getQuaternionFromEuler([0.,-math.pi,math.pi/2.])

        dv = 0.0022     # set panda's base moving distance per step
        dx = action[0] * dv     # panda's moving distance (x coordinate)
        dy = action[1] * dv
        dz = action[2] * dv
        fingers = action[3]     # if fingers == 1 then gripper are released at this step, vice versa

        currentPose = p.getLinkState(self.pandaUid, 11)     # end-effector's current position + orientation
        self.currentPosition = currentPose[0]               # end-effector's current position
        # print('current pose : ', currentPosition)
        newPosition = [self.currentPosition[0] + dx,        # new position to move at this step
                       self.currentPosition[1] + dy,
                       self.currentPosition[2] + dz]
        jointPoses = p.calculateInverseKinematics(self.pandaUid,11,newPosition, orientation)[0:7]

        p.setJointMotorControlArray(self.pandaUid, list(range(7))+[9,10], p.POSITION_CONTROL, list(jointPoses)+2*[fingers])
        # move Panda's joints to newPosition
        p.stepSimulation()
        state_goal_object, _ = p.getBasePositionAndOrientation(self.objectUid[self.duck])

        local_goal_object = self.objectUid[self.duck]   # local goal object can be a lego or the tray(at process 1 and 2)
        if process == 1 or process == 2 : # move to tray
            local_goal_object = self.trayUid

        state_object, _ = p.getBasePositionAndOrientation(local_goal_object) # the position and the orientation of the object
        state_robot = p.getLinkState(self.pandaUid, 11)[0]      # the position of end-effector
        state_fingers = (p.getJointState(self.pandaUid,9)[0], p.getJointState(self.pandaUid, 10)[0])    # each finger's distance to the center

        done = self.processDoneCondition(process, state_goal_object, self.currentPosition)      # returns each process is done or not according to the end-effector's state and the object's state

        self.step_counter += 1

        info = {'object_position': state_object}
        self.panda_position = state_robot + state_fingers       #5-tuple
        print(self.panda_position)
        reward = 0      # not use reward var
        return np.array(self.panda_position).astype(np.float32), reward, done, info

    def processDoneCondition(self, process, state_goal_object, currentPosition):    # Check if each process is done or not according to the object's state and gripper's state
        done = False
        if process == 0 :
            if state_goal_object[2]>0.4: # picked a duck successfully
                done = True
        elif process == 1 :
            if state_goal_object[0]>0.65 : # moved a duck above the tray
                done = True
        elif process == 2 : # release the gripper
            if state_goal_object[0]>0.6 and state_goal_object[2]<0.15 :
                done = True
        elif process == 3 : # put a duck on the tray
            done = True
        elif process == 4 : # initialized panda's pos
            if currentPosition[2]>0.3 and currentPosition[0]<0.4 :
                done = True
        return done

    def reset(self):       # Initialize the environment
        duckcount = 3       #3 objects to move
        rotate = [0, 0.1, 0.6, 0]
        dis = 0.45 / duckcount  # set each object's base location

        self.step_counter = 0
        p.resetSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0) # we will enable rendering after we loaded everything
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0,0,-9.8)

        '''Load Everything we need . . .'''
        self.pandaUid = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
        self.tableUid = p.loadURDF("table/table.urdf", basePosition=[0.5, 0, -0.65])
        self.plainID = p.loadURDF("plane.urdf", basePosition=[0, 0, -0.65])
        self.trayUid = p.loadURDF("tray/traybox.urdf", basePosition=[1., 0, 0])
        self.objectUid= []
        for i in range(duckcount):
            inipos = [0.55 - dis * i, 0, 0]
            self.objectUid.append(p.loadURDF("lego/lego.urdf", basePosition = inipos, baseOrientation= rotate, globalScaling = 1.5))

        '''reset panda's poses'''
        rest_poses = [0, -0.215, 0, -2.57, 0, 2.356, 2.356, 0.08, 0.08]
        for i in range(7):
            p.resetJointState(self.pandaUid, i, rest_poses[i])
        p.resetJointState(self.pandaUid, 9, 0.08)
        p.resetJointState(self.pandaUid, 10, 0.08)

        '''Set class' variables'''
        self.state_robot = p.getLinkState(self.pandaUid, 11)[0]
        self.orien_robot = p.getLinkState(self.pandaUid, 11)[1]
        self.currentPosition = self.state_robot
        state_fingers = (p.getJointState(self.pandaUid,9)[0], p.getJointState(self.pandaUid, 10)[0])
        # print('state_fingers :', state_fingers)
        self.panda_position = self.state_robot + state_fingers
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)

        return np.array(self.panda_position).astype(np.float32)

    def render(self):       # Just camera rendering function

        # view_matrix = p.computeViewMatrixFromYawPitchRoll(    #fixed camera view(view from human)
        #                                                     cameraTargetPosition=[0.7,0,0.05],
        #                                                     distance=.7,
        #                                                     yaw=90,
        #                                                     pitch=-70,
        #                                                     roll=0,
        #                                                     upAxisIndex=2)
        dis = 1000.0
        yaw = p.getEulerFromQuaternion(self.orien_robot)[-1]

        xCamera = self.currentPosition[0]
        yCamera = self.currentPosition[1]
        zCamera = self.currentPosition[2] + 0.15

        xTarget = xCamera + math.cos(yaw) * dis
        yTarget = yCamera + math.sin(yaw) * dis
        zTarget = zCamera + math.tan(yaw) * dis

        view_matrix = p.computeViewMatrix(cameraEyePosition= [xCamera, yCamera, zCamera],
                                          cameraTargetPosition = [xTarget,yTarget,zTarget],
                                          cameraUpVector = [0., 0., 0.5]
                                          )
        proj_matrix = p.computeProjectionMatrixFOV(fov=80,
                                                     aspect=float(960) /720,
                                                     nearVal=0.1,
                                                     farVal=100.0)
        (_, _, px, _, _) = p.getCameraImage(width=960,
                                              height=720,
                                              viewMatrix=view_matrix,
                                              projectionMatrix=proj_matrix,
                                              renderer=p.ER_BULLET_HARDWARE_OPENGL)

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (720,960, 4))

        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def _get_state(self):
        return self.panda_position

    def close(self):
        p.disconnect()
