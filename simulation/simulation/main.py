from panda_env import PandaEnv
import pybullet as p
import math
import click

def launch(obj1_pos, obj1_ori, obj2_pos, obj2_ori) :
    env = PandaEnv()
    done = False
    error = 0.01
    dt = 1. / 240.  # the default timestep in pybullet is 240 Hz

    start = env.reset(obj1_pos=obj1_pos, obj1_ori=obj1_ori, obj2_pos=obj2_pos, obj2_ori=obj2_ori)
    sum_reward = 0
    search_height = 0.2
    env.debugparameter(start)

    for t in range(10000):
        '''gave arbitrary goal poses'''
        print('t : ', t)
        goal_pos = (1., 0.1, search_height)
        goal_orien = p.getQuaternionFromEuler([0, -math.pi/2, math.pi/2])

        # v_x, v_y, v_theta= POMCPOW(observation)
        # goal_pos, goal_orien = getgoalstate(v_x, v_y, v_theta, search_height)

        fingers = 1
        panda_position, reward, done, observation = env.step(goal_pos, goal_orien, fingers)
        sum_reward += reward

    env.close()

@click.command()
@click.option('--obj1_pos', type=tuple, default=(0, 0, 0),
              help='(x, y, z=0.08) : left-most corner of 1st object. Set 0<x<1.3, 0.4<y<0.6, z=0.08')
@click.option('--obj1_ori', type=float, default=0.,
              help='theta : orientation of the 1st object. Set 0<theta<pi/8')
@click.option('--obj2_pos', type=tuple, default=(0, 0, 0),
              help='(x, y, z=0.08) : left-most corner of 2nd object. Set 0<x<1.3, 0.4<y<0.6, z=0.08')
@click.option('--obj2_ori', type=float, default=0.,
              help='theta : orientation of the 2nd object. Set 0<theta<pi/8')

def main(**kwargs) :
    launch(**kwargs)

if __name__ == "__main__":
    main()
