from isaacgym import gymapi
import torch
import os
import abc



class POMDPBaseTask(abc.ABC):

    def __init__(self):

        # set isaac gym    
        self.gym = gymapi.acquire_gym()
    
        # Configure vectask
        sim_params = gymapi.SimParams()
        sim_params.dt = 0.01
        sim_params.physx.use_gpu = True
        sim_params.use_gpu_pipeline = True
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
        self.sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

        # ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up
        plane_params.distance = 0
        plane_params.static_friction = 1
        plane_params.dynamic_friction = 1
        plane_params.restitution = 0
        self.gym.add_ground(self.sim, plane_params)

        # prepare assets
        asset_root = os.path.join("/home/ssh199898/workspace/isaacgym/assets")
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.armature = 0.01
        asset_options.flip_visual_attachments = True
        self.franka_asset = self.gym.load_asset(self.sim, asset_root, "urdf/franka_description/robots/franka_panda.urdf", asset_options)
        self.cabinet_asset = self.gym.load_asset(self.sim, "./assets/", "urdf/cabinet.urdf", asset_options)

        # set up the env grid with auto spacing
        self.num_envs = 64
        envs_per_row = 8
        env_spacing = 2.0
        env_lower = gymapi.Vec3(0, 0.0, -env_spacing)
        env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)

        # caching common handles
        self.envs = []
        self.franka_actor_handles = []
        self.cabinet_actor_handles = []

        for i in range(self.num_envs):

            # creating an environment
            env = self.gym.create_env(self.sim, env_lower, env_upper, envs_per_row)
            self.envs.append(env)

            # creating a franka actor
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
            franka_actor_handle = self.gym.create_actor(env, self.franka_asset, pose, "FrankaPanda", i, 1)
            self.franka_actor_handles.append(franka_actor_handle)

            # creating a cabinet actor
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(0.7, 0.0, 0.0)
            franka_actor_handle = self.gym.create_actor(env, self.cabinet_asset, pose, "Cabinet", i, 1)
            self.cabinet_actor_handles.append(franka_actor_handle)


            # Tips
            ''' 
            Once you finish populating one environment 
            and start populating the next one, 
            you can no longer add actors to the 
            previous environment.
            '''

        # prepare simulation buffers and tensor storage 
        self.gym.prepare_sim(self.sim)

        # creating viewer (comment this for headless mode)
        cam_props = gymapi.CameraProperties()
        self.viewer = self.gym.create_viewer(self.sim, cam_props)


    def __del__(self):
        # clean up
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)



    @abc.abstractmethod
    def reset_idx(self, ids):
        """
        reset selected environments
        - param: ids
        """
    

    @abc.abstractmethod
    def pre_physics_step(self, action: torch.Tensor):
        """
        Compute actions
        - param: action
        """


    @abc.abstractmethod
    def post_physics_step(self) -> torch.Tensor:
        """
        Compute observations, rewards, and resets
        - return: observation
        """
        

    def reset(self):
        for i in range(self.num_envs):
            self.reset_idx(i)


    def step(self, action=None) -> torch.Tensor:
        self.pre_physics_step(action)
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        observation = self.post_physics_step()

        return observation


    def render(self):
        # update the viewer
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, True)

        # Wait for dt to elapse in real time.
        # This synchronizes the physics simulation with the rendering rate.
        self.gym.sync_frame_time(self.sim)

