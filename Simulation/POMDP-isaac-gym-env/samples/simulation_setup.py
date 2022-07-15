from isaacgym import gymapi     # core API
from isaacgym import gymtorch   # PyTorch interop

import os
import math
import random


# set isaac gym
gym = gymapi.acquire_gym()


# create a new sim
sim_params = gymapi.SimParams()
sim_params.dt = 0.01
sim_params.physx.use_gpu = True
sim_params.use_gpu_pipeline = True
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)


# ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up
plane_params.distance = 0
plane_params.static_friction = 1
plane_params.dynamic_friction = 1
plane_params.restitution = 0
gym.add_ground(sim, plane_params)


# prepare assets
asset_root = os.path.join("/home/ssh199898/workspace/isaacgym/assets")
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.armature = 0.01
asset_options.flip_visual_attachments = True
franka_asset = gym.load_asset(sim, asset_root, "urdf/franka_description/robots/franka_panda.urdf", asset_options)


# set up the env grid with auto spacing
num_envs = 64
envs_per_row = 8
env_spacing = 2.0
env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)


# caching common handles
envs = []
actor_handles = []


for i in range(num_envs):
    # creating an environment
    env = gym.create_env(sim, env_lower, env_upper, envs_per_row)
    envs.append(env)

    # creating an actor
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0.0, 10.0)
    actor_handle = gym.create_actor(env, franka_asset, pose, "MyActor", i, 1)
    actor_handles.append(actor_handle)

    # Tips
    ''' 
    Once you finish populating one environment 
    and start populating the next one, 
    you can no longer add actors to the 
    previous environment.
    '''


# prepare simulation buffers and tensor storage 
gym.prepare_sim(sim)


# creating viewer (comment this for headless mode)
cam_props = gymapi.CameraProperties()
viewer = gym.create_viewer(sim, cam_props)


while True:
    
    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)


# clean up
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)