"""
POMDP modeling about SLAG(simultaneously Localizing and Grasping) problem
"""


import os
from POMDP_framework import *
from POMCP import *
from POMCPOW import *
import util

import random
import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib import animation

import time
import pickle

from policy import Settings, NNRegressionPolicyModel


class State(State):
    """
    The state of the problem is the difference pose of two objects with end-effector in 2D plain (x_1,y_1,theta_1,x_2,y_2,theta_2,w_1,w_2).
    """
    def __init__(self, state):
        """
        Initializes a state.

        Args:
            state (tuple): pose of two objects.
        """
        if len(state) != 8:
            raise ValueError("State position must be a vector of length 8")
        self.state = state

    def __hash__(self):
        return hash(tuple(self.state))
    
    def __eq__(self, other):
        if isinstance(other, State):
            return self.state == other.state
        else:
            return False
        
    def __str__(self):
        return self.__repr__()
    
    def __repr__(self):
        return "State(%s)" % (str(self.position))


class Action(Action):
    """
    The action is a vector of velocities in 2D(v_x,v_y,v_theta).
    """
    def __init__(self, action):
        """
        Initializes a action.

        Args:
            action (tuple): velocity
        """
        if len(action) != 3:
            raise ValueError("Action control must be a vector of length 3")        
        self.action = action

    def __hash__(self):
        return hash(self.action)
    
    def __eq__(self, other):
        if isinstance(other, Action):
            return self.action == other.action
        else:
            return False
        
    def __str__(self):
        return self.__repr__()
    
    def __repr__(self):
        return "Action(%s)" % (str(self.action))


class Observation(Observation):
    """
    Defines the Observation for the continuous SLAG domain;
    Observation space: 
        The distance between end-effector and the object in front, adding Guassian noise.
    """
    
    def __init__(self, observation):
        """
        Initializes a observation.

        Args:
            observation (float): distance
        """
        self.observation = observation

    def __eq__(self, other):
        if isinstance(other, Observation):
            return self.observation == other.observation
        else:
            return False
        
    def __str__(self):
        return self.__repr__()
    
    def __repr__(self):
        return "Observation(%s)" % (str(self.observation))


class TransitionModel(TransitionModel):
    """
    The underlying deterministic system dynamics
    """
    def __init__(self, epsilon=1e-9):
        self._epsilon = epsilon

    def probability(self, next_state, state, action, **kwargs):
        """
        Deterministic.
        """
        expected_pose = tuple(self.func(state.state, action))
        if next_state.pose == expected_pose:
            return 1.0 - self._epsilon
        else:
            return self._epsilon

    def sample(self, state, action):
        next_state = copy.deepcopy(state)
        next_state.state = tuple(self.func(state.state, action))
        return next_state

    def argmax(self, state, action):
        """Returns the most likely next state"""
        return self.sample(state, action)

    #|FIXME| Is the corner for state invariant? Can the orientation be negative?
    def func(self, state_pos, action):
        """
        Returns the function of the underlying system dynamics.
        The function is: (xt, ut) -> xt+1 where xt, ut, xt+1 are all numpy arrays.
        """
        return np.array([state_pos[0] - action[0],
                         state_pos[1] - action[1],
                         state_pos[2] - action[2],
                         state_pos[3] - action[0],
                         state_pos[4] - action[1],
                         state_pos[5] - action[2],
                         state_pos[6],
                         state_pos[7]])


class ObservationModel(ObservationModel):

    # def __init__(self, light, const):
    #     """
    #     `light` and `const` are parameters in
    #     :math:`w(x) = \frac{1}{2}(\text{light}-s_x)^2 + \text{const}`

    #     They should both be floats. The quantity :math:`w(x)` will
    #     be used as the variance of the covariance matrix in the gaussian
    #     distribution (this is how I understood the paper).
    #     """
    #     self._light = light
    #     self._const = const

    # def _compute_variance(self, pos):
    #     return 0.5 * (self._light - pos[0])**2 + self._const

    # def noise_covariance(self, pos):
    #     variance = self._compute_variance(pos)
    #     return np.array([[variance, 0],
    #                      [0, variance]])

    # # |FIXME| observe according to true/belief state??
    # def probability(self, observation, next_true_state, next_belief, action):
    #     """
    #     The observation is :math:`g(x_t) = x_t+\omega`. So
    #     the probability of this observation is the probability
    #     of :math:`\omega` which follows the Gaussian distribution.
    #     """
    #     # if self._discrete:
    #     #     observation = observation.discretize()
    #     variance = self._compute_variance(next_true_state.position)
    #     gaussian_noise = Gaussian([0,0],
    #                               [[variance, 0],
    #                                [0, variance]])
    #     omega = (observation.position[0] - next_belief.position[0],
    #              observation.position[1] - next_belief.position[1])
    #     return gaussian_noise[omega]

    # def sample(self, next_state, action, argmax=False):
    #     """sample an observation."""
    #     # Sample a position shift according to the gaussian noise.
    #     obs_pos = self.func(next_state.position, False)
    #     return Observation(tuple(obs_pos))
        
    # def argmax(self, next_state, action):
    #     return self.sample(next_state, action, argmax=True)

    # def func(self, next_state_pos, mpe=False):
    #     variance = self._compute_variance(next_state_pos)
    #     gaussian_noise = Gaussian([0,0],
    #                               [[variance, 0],
    #                                [0, variance]])
    #     if mpe:
    #         omega = gaussian_noise.mpe()
    #     else:
    #         omega = gaussian_noise.random()
    #     return np.array([next_state_pos[0] + omega[0],
    #                      next_state_pos[1] + omega[1]])

    # def func_noise(self):
    #     """Returns a function that returns a state-dependent Gaussian noise."""
    #     def fn(mt):
    #         variance = self._compute_variance(mt)
    #         gaussian_noise = Gaussian([0,0],
    #                                   [[variance, 0],
    #                                    [0, variance]])
    #         return gaussian_noise
    #     return fn


class RewardModel(RewardModel):
    # def __init__(self, light, goal_state, epsilon):
    #     self.light = light
    #     self._goal_state = goal_state
    #     self._epsilon=epsilon

    # def _reward_func_state(self, state: State, action, next_state: State, goal_state: State, epsilon):
    #     if np.sum((np.asarray(goal_state.position) - np.asarray(next_state.position))**2) < epsilon**2:
    #         reward = 100
    #     else:
    #         # reward = (-1) * np.abs(next_state.position[0] - self.light)
    #         reward = -1
    #     return reward

    #     # # Euclidean distance
    #     # reward = (-1)*np.sum((np.asarray(goal_state.position) - np.asarray(next_state.position))**2)
    #     # return reward

    # def _reward_func_particle(self, state: Particles, action: Action, next_state: Particles, goal_state: State, epsilon: float):
    #     # for belief state - be used to real action
    #     reward_expectatation = 0
    #     for state_in in next_state.particles:
    #         reward_expectatation += self._reward_func_state(state_in, action, state_in, self._goal_state, self._epsilon)
    #     return reward_expectatation / len(next_state)

    # def _reward_func_hist(self, state: Histogram, action: Action, next_state: Histogram, goal_state: State, epsilon: float):
    #     # for belief state - be used to real action
    #     normalized_hist = next_state.get_normalized()
    #     reward_expectatation = 0
    #     for state_in in normalized_hist:
    #         reward_expectatation += normalized_hist[state_in] * self._reward_func_state(state_in, action, state_in, self._goal_state, self._epsilon)
    #     return reward_expectatation

    # def sample(self, state, action, next_state):
    #     # |TODO| make exception
    #     # deterministic      
    #     if str(type(next_state)) == "<class '__main__.State'>":
    #         return self._reward_func_state(state, action, next_state, self._goal_state, self._epsilon)
    #     # |TODO| currently, get reward after update, can make same as state case(before update)?
    #     elif str(type(next_state)) == "<class 'POMDP_framework.Particles'>":
    #         return self._reward_func_particle(state, action, next_state, self._goal_state, self._epsilon)
    #     elif str(type(next_state)) == "<class 'POMDP_framework.Histogram'>":
    #         return self._reward_func_hist(state, action, next_state, self._goal_state, self._epsilon)
        
    # # For State
    # def is_goal_state(self, state: State):
    #     if np.sum((np.asarray(self._goal_state.position) - np.asarray(state.position))**2) < self._epsilon**2:
    #         return True
    #     return False

    # # For Histogram
    # def is_goal_hist(self, state: Histogram, thres=0.7):
    #     # test goal condition: #particle(prob) in goal_state >= thres -> True
    #     prob_in_goal = 0
    #     normalized_hist = state.get_normalized()
    #     for particle in normalized_hist:
    #         if np.sum((np.asarray(self._goal_state.position) - np.asarray(particle.position))**2) < self._epsilon**2:
    #             prob_in_goal += normalized_hist[particle]
    #     print(r"% of particles in goal: " + str(prob_in_goal*100) + "%")
    #     return prob_in_goal >= thres
    
    # # For Particles
    # def is_goal_particles(self, state: Particles, thres=0.7):
    #     # test goal condition: #particle(prob) in goal_state >= thres -> True
    #     num_particles = len(state)
    #     prob_in_goal = 0
    #     for particle in state.particles:
    #         if np.sum((np.asarray(self._goal_state.position) - np.asarray(particle.position))**2) < self._epsilon**2:
    #             prob_in_goal += 1
    #     prob_in_goal /= num_particles
    #     print(r"% of particles in goal: " + str(prob_in_goal*100) + "%")
    #     return prob_in_goal >= thres


class PolicyModel(RandomRollout):
    # def _NextAction(self, state, x_range=(-1,6), y_range=(-1,6)):
    #     pos = state.position
    #     _action_x = random.uniform(x_range[0] - pos[0], x_range[1] - pos[0])
    #     _action_y = random.uniform(y_range[0] - pos[1], y_range[1] - pos[1])
    #     _action = (_action_x,_action_y)
    #     return _action


class LightDarkEnvironment(Environment):
    def __init__(self,
                 init_state,
                 light,
                 const,
                 reward_model=None):
        """
        Args:
            init_state (light_dark.domain.State or np.ndarray):
                initial true state of the light-dark domain,
            goal_pos (tuple): goal position (x,y)
            light (float):  see below
            const (float): see below
            reward_model (pomdp_py.RewardModel): A reward model used to evaluate a policy
        `light` and `const` are parameters in
        :math:`w(x) = \frac{1}{2}(\text{light}-s_x)^2 + \text{const}`

        Basically, there is "light" at the x location at `light`,
        and the farther you are from it, the darker it is.
        """
        self._light = light
        self._const = const
        self.init_state = init_state
        transition_model = TransitionModel()
        if type(init_state) == np.ndarray:
            init_state = State(init_state)
        super().__init__(init_state,
                         transition_model,
                         reward_model)

    @property
    def light(self):
        return self._light

    @property
    def const(self):
        return self._const


class LightDarkProblem(POMDP):
    def __init__(self, init_state, init_belief, goal_state, light, const, epsilon, guide_policy=None):
        if guide_policy is not None:
            # nn_config = Settings()
            # guide_policy = NNRegressionPolicyModel(nn_config)
            agent = Agent(init_belief,
                      guide_policy,
                      TransitionModel(),
                      ObservationModel(light,const),
                      RewardModel(light, goal_state, epsilon))
        else:
            agent = Agent(init_belief,
                        PolicyModel(),
                        TransitionModel(),
                        ObservationModel(light,const),
                        RewardModel(light, goal_state, epsilon))

        env = LightDarkEnvironment(init_state,                  # init state
                                   light,                       # light
                                   const,                       # const
                                   RewardModel(light, goal_state, epsilon))     # reward model
        
        self.goal_state = goal_state
        
        super().__init__(agent, env, name="LightDarkProblem")


def expectation_belief(belief: Particles):
    # # For Histogram
    # total_weight = 0
    # weighted_sum = [0,0]
    # for state in hist:
    #     total_weight += hist[state]
    #     pos = state.position
    #     weighted_sum[0] += hist[state]*pos[0]
    #     weighted_sum[1] += hist[state]*pos[1]
    # weighted_sum[0] /= total_weight
    # weighted_sum[1] /= total_weight
    
    # For Particles
    num_particles = len(belief)
    expectation = [0, 0]
    for p in belief.particles:
        pos = p.position
        expectation[0] += pos[0]
        expectation[1] += pos[1]
    expectation[0] /= num_particles
    expectation[1] /= num_particles
    
    return expectation


def main():
    plotting = None
    save_log = False
    save_data = False
    # save_sim_data = 'sim_success_10'
    save_sim_data = False
    # name_dataset = 'mcts_3_'
    name_dataset = None
    exp = False

    guide = False
    rollout_guide = False

    # Environment Setting
    # define the observation noise equation.
    light = 5.0
    const = 0.001
    # define the radius of goal region
    epsilon = 0.25

    # planning horizon
    planning_horizon = 30

    # defines discount_factor
    discont_factor = 1.0

    num_sucess = 0
    num_fail = 0
    num_planning = 1
    num_particles = 100

    if save_data:
        save_dir = os.path.join(os.getcwd(),'Learning/dataset', 'mcts_')
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
            
    if save_sim_data:
        save_dir_sim = os.path.join(os.getcwd(),'Learning/dataset', 'sim_success_2.7_exp_const_30_std0.5')
        if not os.path.exists(save_dir_sim):
            os.mkdir(save_dir_sim)
    else:
        save_dir_sim = False

    if exp:
        log_dir = os.path.join(os.getcwd(), 'result/log')
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

    log_time = []
    log_total_reward = []
    log_total_reward_success = []
    log_total_reward_fail = []
    log_sim_success_rate = []
    log_val_root_success_traj_avg = []
    log_val_action_success_traj_avg = []
    log_val_root_fail_traj_avg = []
    log_val_action_fail_traj_avg = []
    log_val_root_step = []
    log_val_action_step = []
    # log_sim_success_rate_step1 = []

    if guide:
        nn_config = Settings()
        guide_policy = NNRegressionPolicyModel(nn_config)
    else:
        guide_policy = None

    for n in range(num_planning):
        print("========================================================") 
        print("========================= %d-th ========================" % (n+1)) 
        print("========================================================")

        if save_sim_data:
            save_sim_name = f'{save_sim_data}_{n}'
        else:
            save_sim_name = False

        # fixed inital & goal state
        init_pos = (2.5, 2.5)
        goal_pos = (0, 0)

        # # randomize initial & goal state
        # init_pos = np.random.rand(2) * 3
        # init_pos[1] += 2
        # goal_pos = np.random.rand(2) - 0.5

        init_state = State(tuple(init_pos))
        goal_state = State(tuple(goal_pos))

        # inital belief state is uniformly distribution
        # init_belief = Histogram({})
        init_belief = []
        # # For uniform initialization
        # while len(init_belief) < num_particles:
        #     sample = State(tuple(2.5 + random_range * (np.random.rand(2)-0.5)))
        #     init_belief.append(sample)
            # init_belief[sample] = 1 / (random_range**2 * num_particles)
        # For gaussian initalization
        # init_belief_std = 1.0
        init_belief_std = 0.5
        while len(init_belief) < num_particles:
            # # uniform distribution
            # radius = init_belief_std * np.random.rand(1)
            # theta = 2 * np.pi * np.random.rand(1)
            # noise = np.asarray([radius * np.cos(theta), radius * np.sin(theta)]).reshape(2)
            # sample = State(tuple(init_state.position + noise))

            # normal distribution
            sample = State(tuple(np.asarray(init_state.position) + init_belief_std * (np.random.randn(2))))

            init_belief.append(sample)

        init_belief = Particles(init_belief)
               
        # creates POMDP model
        light_dark_problem = LightDarkProblem(init_state, init_belief, goal_state, light, const, epsilon, guide_policy)
        # light_dark_problem.agent.set_belief(Particles.from_histogram(init_belief,num_particles=1))
        light_dark_problem.agent.set_belief(init_belief)

        # set planner
        # planner = POMCPOW(pomdp=light_dark_problem, max_depth=planning_horizon, planning_time=-1., num_sims=num_particles,
        #                 discount_factor=discont_factor, save_dir_sim=save_dir_sim, exploration_const=math.sqrt(2),
        #                 num_visits_init=0, value_init=0)
        planner = POMCPOW(pomdp=light_dark_problem, max_depth=planning_horizon, planning_time=-1., num_sims=num_particles,
                        discount_factor=discont_factor, save_dir_sim=save_dir_sim, exploration_const=30,
                        num_visits_init=0, value_init=0)

        # Visualization setting
        if plotting:
            x_range = (-1, 6)
            y_range = (-1, 6)
            viz = LightDarkViz(light_dark_problem, x_range, y_range, 0.1)

        # planning
        print("==== Planning ====")
        total_reward = 0
        total_num_sims = 0
        total_num_sims_success = 0
        total_plan_time = 0.0
        log_time_each = []
        log_val_each_root = []
        log_val_each_action = []
        traj_data = []

        for i in range(planning_horizon):
            logging = save_log and i==0
            if i == 0:
                print("Goal state: %s" % goal_state)
                print("Inital state: %s" % light_dark_problem.env.state)
                print("Inital belief state expectation:", expectation_belief(light_dark_problem.agent.cur_belief))
                # print("Inital belief state: %s" % str(light_dark_problem.agent.cur_belief))
                print("Number of particles:", len(light_dark_problem.agent.cur_belief))

                # initial history
                init_observation = light_dark_problem.agent._observation_model.sample(light_dark_problem.env.state, (0,0))
                light_dark_problem.agent.update_history((0,0), init_observation.position, light_dark_problem.env.state.position, 0)

                if plotting:
                    viz.log_state(light_dark_problem.env.state)
                    viz.log_belief_expectation(expectation_belief(light_dark_problem.agent.cur_belief))

            print("==== Step %d ====" % (i+1))

            if save_sim_name:
                save_sim_name_ = f'{save_sim_name}_{i}'
            else:
                save_sim_name_ = False
            best_action, time_taken, sims_count, sims_count_success, root_value, action_value, step_data = planner.plan(light_dark_problem.agent, i, logging, save_sim_name_, guide, rollout_guide)
            
            traj_data.append(step_data)
            log_time_each.append(time_taken)
            log_val_each_root.append(root_value)
            log_val_each_action.append(action_value)
            log_time_each_avg = np.mean(np.asarray(log_time_each))
            log_val_each_root_avg = np.mean(np.asarray(log_val_each_root))
            log_val_each_action_avg = np.mean(np.asarray(log_val_each_action))

            # |FIXME|
            next_state = light_dark_problem.agent.transition_model.sample(light_dark_problem.env.state, best_action)
            real_observation = light_dark_problem.agent.observation_model.sample(next_state, best_action)
            
            # select observataion node in existing node - unrealistic
            # real_observation = random.choice(list(planner._agent.tree[best_action].children.keys()))
            
            total_num_sims += sims_count
            total_num_sims_success += sims_count_success
            total_plan_time += time_taken

            check_goal = planner.update(light_dark_problem.agent, light_dark_problem.env, best_action, next_state, real_observation)
            # |TODO| can move before update to avoid confusion state case and belief case?
            # By belief state
            # reward = light_dark_problem.env.reward_model.sample(light_dark_problem.agent.cur_belief, best_action, light_dark_problem.agent.cur_belief)
            # # By true state
            # reward = light_dark_problem.env.reward_model.sample(next_state, best_action, next_state)

            # |NOTE| only take positive reward as achieving goal condition
            if not check_goal: # if you want to use reward proportional to the number of particles which is satisfied the goal condition, use reward_model.sample().
                reward = -1.
            else:
                reward = 100.

            total_reward = reward + discont_factor * total_reward

            # |TODO| how to move in planner.update? need to resolve "TODO" for reward
            # update history
            light_dark_problem.agent.update_history(best_action, real_observation.position, next_state.position, reward)

            print("Action: %s" % str(best_action))
            print("Observation: %s" % real_observation)
            print("Goal state: %s" % goal_state)
            print("True state: %s" % light_dark_problem.env.state)
            print("Belief state expectation:", expectation_belief(light_dark_problem.agent.cur_belief))
            # print("Belief state: %s" % str(light_dark_problem.agent.cur_belief))
            print("Number of particles:", len(light_dark_problem.agent.cur_belief))
            print("Reward: %s" % str(reward))
            print("Num sims: %d" % sims_count)
            print("Num sims success: %d" % sims_count_success)
            # if i == 0:
            #     log_sim_success_rate_step1.append(sims_count_success/sims_count)
            print("Plan time: %.5f" % time_taken)
            
            if plotting:
                # viz.set_initial_belief_pos(b_0[0])
                # viz.log_position(tuple(b_0[0]), path=0)
                # viz.log_position(tuple(b_0[0]), path=1)
                viz.log_state(light_dark_problem.env.state)
                viz.log_belief(light_dark_problem.agent.cur_belief)
                viz.log_belief_expectation(expectation_belief(light_dark_problem.agent.cur_belief))

            if check_goal:
                print("\n")
                print("==== Success ====")
                print("Total reward: %.5f" % total_reward)
                log_total_reward.append(total_reward)
                log_total_reward_success.append(total_reward)
                # print("History:", planner.history)
                print("Total Num sims: %d" % total_num_sims)
                print(f"Num sims success: {total_num_sims_success} ({100 * total_num_sims_success/total_num_sims}%)")
                print("Total Plan time: %.5f" % total_plan_time)
                num_sucess += 1
         
                log_val_root_success_traj_avg.append(log_val_each_root_avg)
                log_val_action_success_traj_avg.append(log_val_each_action_avg)
                log_val_root_step.append(log_val_each_root)
                log_val_action_step.append(log_val_each_action)
                
                # # save data
                # if save_data:
                #     with open(os.path.join(save_dir,'success_history.pickle'), 'ab') as f:
                #         pickle.dump(planner.history[:-1], f, pickle.HIGHEST_PROTOCOL)
                #     with open(os.path.join(save_dir,'success_value.pickle'), 'ab') as f:
                #         pickle.dump(total_reward, f, pickle.HIGHEST_PROTOCOL)
                
                # # Saving success simulation history
                # if save_sim_data:
                #     if planner.history_data is None:
                #         pass
                #     else:
                #         sim_data = planner.history_data
                #         sim_data.append(goal_pos)
                #         sim_data.append(total_reward)
                #         with open(os.path.join(save_dir_sim, 'simulation_history_data.pickle'), 'ab') as f:
                #             pickle.dump(sim_data, f)
                        
                if save_data:
                    traj_data.append(goal_pos)
                    traj_data.append(total_reward)
                    with open(os.path.join(save_dir, f'{name_dataset}_{n}.pickle'), 'wb') as f:
                        pickle.dump(traj_data, f)
                
                break

            elif i == planning_horizon-1:
                print("==== Fail ====")
                print("Total reward: %.5f" % total_reward)
                log_total_reward.append(total_reward)
                log_total_reward_fail.append(total_reward)
                # print("History:", planner.history)
                print("Total Num sims: %d" % total_num_sims)
                print(f"Num sims success: {total_num_sims_success} ({100 * total_num_sims_success/total_num_sims}%)")
                print("Total Plan time: %.5f" % total_plan_time)
                num_fail += 1

                log_val_root_fail_traj_avg.append(log_val_each_root_avg)
                log_val_action_fail_traj_avg.append(log_val_each_action_avg)
                log_val_root_step.append(log_val_each_root)
                log_val_action_step.append(log_val_each_action)

                # # save data
                # if save_data:
                #     with open(os.path.join(save_dir,'fail_history.pickle'), 'ab') as f:
                #         pickle.dump(planner.history[:-1], f, pickle.HIGHEST_PROTOCOL)
                #     with open(os.path.join(save_dir,'fail_value.pickle'), 'ab') as f:
                #         pickle.dump(total_reward, f, pickle.HIGHEST_PROTOCOL)

                # # Saving fail history
                # if save_sim_data:
                #     if planner.history_data is None:
                #         pass
                #     else:
                #         sim_data = planner.history_data
                #         sim_data.append(goal_pos)
                #         sim_data.append(total_reward)
                #         with open(os.path.join(save_dir_sim, 'simulation_history_data.pickle'), 'ab') as f:
                #             pickle.dump(sim_data, f)
                        
                if save_data:
                    traj_data.append(goal_pos)
                    traj_data.append(total_reward)
                    with open(os.path.join(save_dir, f'{name_dataset}_{n}.pickle'), 'wb') as f:
                        pickle.dump(traj_data, f)
            
        if plotting is not None:
            viz.plot(path_colors={0: [(0,0,0), (0,255,0)],
                                    1: [(0,0,0), (255,0,0)]},
                        path_styles={0: "--",
                                    1: "-"},
                        path_widths={0: 4,
                                    1: 1})
            plt.show()
            # plt.savefig(f'{plotting}.png')

        log_time.append(log_time_each_avg)
        log_sim_success_rate.append(total_num_sims_success/total_num_sims)

        print("num_sucess: %d" % num_sucess)
        print("num_fail: %d" % num_fail)

    sims_success_rate = np.mean(np.asarray(log_sim_success_rate))
    time_mean = np.mean(np.asarray(log_time))
    time_std = np.std(np.asarray(log_time))
    total_reward_mean = np.mean(np.asarray(log_total_reward))
    total_reward_std = np.std(np.asarray(log_total_reward))
    total_reward_success_mean = np.mean(np.asarray(log_total_reward_success))
    total_reward_success_std = np.std(np.asarray(log_total_reward_success))
    total_reward_fail_mean = np.mean(np.asarray(log_total_reward_fail))
    total_reward_fail_std = np.std(np.asarray(log_total_reward_fail))
    root_val_total = log_val_root_success_traj_avg + log_val_root_fail_traj_avg
    action_val_total = log_val_action_success_traj_avg + log_val_action_fail_traj_avg
    root_val_total_mean = np.mean(np.asarray(root_val_total))
    action_val_total_mean = np.mean(np.asarray(action_val_total))
    root_val_total_std = np.std(np.asarray(root_val_total))
    action_val_total_std = np.std(np.asarray(action_val_total))
    root_val_success_mean = np.mean(np.asarray(log_val_root_success_traj_avg))
    action_val_success_mean = np.mean(np.asarray(log_val_action_success_traj_avg))
    root_val_success_std = np.std(np.asarray(log_val_root_success_traj_avg))
    action_val_success_std = np.std(np.asarray(log_val_action_success_traj_avg))
    root_val_fail_mean = np.mean(np.asarray(log_val_root_fail_traj_avg))
    action_val_fail_mean = np.mean(np.asarray(log_val_action_fail_traj_avg))
    root_val_fail_std = np.std(np.asarray(log_val_root_fail_traj_avg))
    action_val_fail_std = np.std(np.asarray(log_val_action_fail_traj_avg))

    # val_step_1_2 = []
    # for v in log_val_step:
    #     val_step_1_2.append(v[0:2])
    # val_step_1_2_mean = np.mean(np.asarray(val_step_1_2).reshape(-1, 2), axis=0)
    # val_step_1_2_std = np.std(np.asarray(val_step_1_2).reshape(-1, 2), axis=0)
    # sims_success_rate_step1 = np.mean(np.asarray(log_sim_success_rate_step1))

    print("====Finish===")
    print("total_num_sucess: %d" % num_sucess)
    print("total_num_fail: %d" % num_fail)
    print(f"sims success rate: {sims_success_rate * 100}%")
    print("planning time(mean): %f" % time_mean)
    print("planning time(std): %f" % time_std)
    print("total reward(mean): %f" % total_reward_mean)
    print("total reward(std): %f" % total_reward_std)
    print("total reward(success/mean): %f" % total_reward_success_mean)
    print("total reward(success/std): %f" % total_reward_success_std)
    print("total reward(fail/mean): %f" % total_reward_fail_mean)
    print("total reward(fail/std): %f" % total_reward_fail_std)
    print("value of tree(total/mean): %f" % root_val_total_mean)
    print("value of tree(total/std): %f" % root_val_total_std)
    print("value of tree(success/mean): %f" % root_val_success_mean)
    print("value of tree(success/std): %f" % root_val_success_std)
    print("value of tree(fail/mean): %f" % root_val_fail_mean)
    print("value of tree(fail/std): %f" % root_val_fail_std)
    print("value of action(total/mean): %f" % action_val_total_mean)
    print("value of action(total/std): %f" % action_val_total_std)
    print("value of action(success/mean): %f" % action_val_success_mean)
    print("value of action(success/std): %f" % action_val_success_std)
    print("value of action(fail/mean): %f" % action_val_fail_mean)
    print("value of action(fail/std): %f" % action_val_fail_std)
    # print("value of tree(mean@step 1,2):", val_step_1_2_mean)
    # print("value of tree(std@step 1,2):", val_step_1_2_std)
    # print(f"sims success rate(@step 1): {sims_success_rate_step1 * 100}%")
    
    if exp:
        with open(os.path.join(log_dir, f'{exp}.txt'), 'w') as f:
            f.write("total_num_sucess: %d\n" % num_sucess)
            f.write("total_num_fail: %d\n" % num_fail)
            f.write(f"sims success rate: {sims_success_rate * 100}%\n")
            f.write("planning time(mean): %f\n" % time_mean)
            f.write("planning time(std): %f\n" % time_std)
            f.write("total reward(mean): %f" % total_reward_mean)
            f.write("total reward(std): %f" % total_reward_std)
            f.write("total reward(success/mean): %f" % total_reward_success_mean)
            f.write("total reward(success/std): %f" % total_reward_success_std)
            f.write("total reward(fail/mean): %f" % total_reward_fail_mean)
            f.write("total reward(fail/std): %f" % total_reward_fail_std)
            f.write("value of tree(total/mean): %f\n" % root_val_total_mean)
            f.write("value of tree(total/std): %f\n" % root_val_total_std)
            f.write("value of tree(success/mean): %f\n" % root_val_success_mean)
            f.write("value of tree(success/std): %f\n" % root_val_success_std)
            f.write("value of tree(fail/mean): %f\n" % root_val_fail_mean)
            f.write("value of tree(fail/std): %f\n" % root_val_fail_std)
            f.write("value of action(total/mean): %f\n" % action_val_total_mean)
            f.write("value of action(total/std): %f\n" % action_val_total_std)
            f.write("value of action(success/mean): %f\n" % action_val_success_mean)
            f.write("value of action(success/std): %f\n" % action_val_success_std)
            f.write("value of action(fail/mean): %f\n" % action_val_fail_mean)
            f.write("value of action(fail/std): %f\n" % action_val_fail_std)

if __name__ == '__main__':
    main()