from isaacgym import gymapi
from isaacgym import gymtorch
import torch
import numpy as np
from .base_task import POMDPBaseTask



class POMDPSimpleTask(POMDPBaseTask):

    def __init__(self):
        super().__init__()
        
        # kinematic config
        self.franka_default_pose = (
            0.0,        # Joint 0
            -0.6,       # Joint 1
            0.0,        # Joint 2
            -2.57,      # Joint 3
            0.0,        # Joint 4
            3.5,        # Joint 5
            0.75,       # Joint 6
            0.08,       # Joint 7 (right gripper)
            0.08        # Joint 8 (left gripper)
        )


        # configure joint properties of the actor in all environments
        for i in range(self.num_envs):
            props = self.gym.get_actor_dof_properties(self.envs[i], self.franka_actor_handles[i])
            props["driveMode"].fill(gymapi.DOF_MODE_POS)    # PD controller
            props["stiffness"].fill(1000.0)
            props["damping"].fill(200.0)
            self.gym.set_actor_dof_properties(self.envs[i], self.franka_actor_handles[i], props)


        # acquire state tensor descriptor (GPU acceleration)
        self._dof_states = self.gym.acquire_dof_state_tensor(self.sim)
        '''
        Tip:
        State tensor shape is contiguous [N, 2], where 
        N = (# of envs) * (# of dofs per env)
        with [0: position, 1: velocity]
        '''


    def pre_physics_step(self, action=None):
        
        # populate latest state from simulation to tensor descriptor
        self.gym.refresh_dof_state_tensor(self.sim)

        # set default pose
        # wrap to pytorch and edit tensor value
        dof_states = gymtorch.wrap_tensor(self._dof_states)
        for i in range(self.num_envs):
            idx = self.gym.get_actor_dof_index(self.envs[i], self.franka_actor_handles[i], 0, gymapi.DOMAIN_SIM)
            dof_states[idx:idx+len(self.franka_default_pose), 0] = torch.Tensor(self.franka_default_pose)
        '''
        Tip:
        Considerable functions for editing the tensor value
        - get_sim_dof_count
        - get_actor_dof_index
        - set_dof_state_tensor
        - set_dof_state_tensor_indexed
        '''

        # apply tensor values to the simulator (pass descriptor, not tensor)
        self.gym.set_dof_state_tensor(self.sim, self._dof_states)
            
        
    def post_physics_step(self):
        return None


    def reset_idx(self, ids):
        pass
