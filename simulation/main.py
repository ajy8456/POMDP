from panda_env import PandaEnv

def process_result(process) : # prints process, returns process are done or not, if done moving an object, process is 4
    if process == 0:    # process 0 : from gripper's initial state to grasp an object and pick up high
        print("--Grasped a duck--")
        return False, process + 1
    if process == 1:    # process 1 : moving an object from left to right(above the box)
        print("--Moving to the box--")
        return False, process + 1
    if process == 2:    # process 2 : move an object from above to the box (still grasping)
        print("--release the duck--")
        return False, process + 1
    if process == 3:    # process 3 : release a finger (put the object down) and move the panda above the box(initialize gripper's pos)
        print("--init panda's pos--")
        return False, process + 1
    if process == 4:    # process 4 == Done Initializing
        return True, 0
    else :
        return False, process

def main() :
    env = PandaEnv()
    done = False
    error = 0.01

    k_p = 3         # set panda's base speed
    k_d = 1
    dt = 1. / 240.  # the default timestep in pybullet is 240 Hz

    panda_position = env.reset()

    for i in range(1, 4) :  # Three objects(lego) to move.
        isEnd = False
        process = 0
        fingers = 1     # fingers var means the distance between two gripper's fingers.
                        # if fingers == 1 then I want to make the gripper to release the object
                        # if fingers == 0 then I want to make the gripper to grasp the object
        object_position = env.duck_position(i)   # Get object's position (x, y, z)

        for t in range(300):    #300 steps per object are available
            env.render()

            target_x = object_position[0]       # target x coordinate(==object x coordinate)
            target_y = object_position[1]
            target_z = object_position[2] - 0.006       # target z coordinate (to grasp surely, target coordinate is a bit below the target coord)

            if (panda_position[3] + panda_position[4]) < error + 0.037 and fingers == 0: # if gripper grasped the object correctly
                target_z = 0.5     # then pick the object
            if process == 2 :   # if done moving the object above the box
                target_z = 0    # then put it down
            elif process == 3 :     # if done moving the object on the box
                fingers = 1     # then release the gripper and put the object in the box
            elif process == 4 :     # initialize panda's pos; moving above and left
                target_z = 0.5
                target_x = 0.3

            '''Calculate a distance to move'''
            dx = target_x - panda_position[0]   # x distance to move
            dy = target_y - panda_position[1]   # y distance to move
            dz = target_z - panda_position[2]   # z distance to move

            if abs(dx) < error and abs(dy) < error and abs(dz) < error: # if panda are ready to grasp the object
                fingers = 0     # make the gripper grasp the object

            '''Calculate a velocity'''
            pd_x = k_p * dx + k_d * dx / dt
            pd_y = k_p * dy + k_d * dy / dt
            pd_z = k_p * dz + k_d * dz / dt

            action = [pd_x, pd_y, pd_z, fingers] # this is the input of the step(is used to calculate panda's next movement)
            # print("action : ", action)
            panda_position, reward, done, info = env.step(action, process, i-1) # not use reward, info variables at now
            object_position = info['object_position']

            if done:    # if a process is done (env.step returns process is done or not)
                isEnd, process = process_result(process)    #if process == 4 and is Done, then isEnd is True
                if isEnd :  # isEnd is True if one object are moved into the box
                    break
                else :
                    continue
    env.close()


if __name__ == "__main__":
    main()
