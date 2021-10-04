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
import time
import pickle

from policy import Settings, NNRegressionPolicyModel


class State(State):
    """The state of the problem is just the robot position"""
    def __init__(self, position):
        """
        Initializes a state in light dark domain.

        Args:
            position (tuple): position of the robot.
        """
        if len(position) != 2:
            raise ValueError("State position must be a vector of length 2")
        self.position = position

    def __hash__(self):
        return hash(tuple(self.position))
    
    def __eq__(self, other):
        if isinstance(other, State):
            return self.position == other.position
        else:
            return False
        
    def __str__(self):
        return self.__repr__()
    
    def __repr__(self):
        return "State(%s)" % (str(self.position))


class Action(Action):
    """The action is a vector of velocities(2-dimension)"""
    def __init__(self, control):
        """
        Initializes a state in light dark domain.

        Args:
            control (tuple): velocity
        """
        if len(control) != 2:
            raise ValueError("Action control must be a vector of length 2")        
        self.control = control

    def __hash__(self):
        return hash(self.control)
    
    def __eq__(self, other):
        if isinstance(other, Action):
            return self.control == other.control
        else:
            return False
        
    def __str__(self):
        return self.__repr__()
    
    def __repr__(self):
        return "Action(%s)" % (str(self.control))


class Observation(Observation):
    """Defines the Observation for the continuous light-dark domain;

    Observation space: 

        :math:`\Omega\subseteq\mathbb{R}^2` the observation of the robot is
            an estimate of the robot position :math:`g(x_t)\in\Omega`.

    """
    # # the number of decimals to round up an observation when it is discrete.
    # PRECISION=2
    
    def __init__(self, position, discrete=False):
        """
        Initializes a observation in light dark domain.

        Args:
            position (tuple): position of the robot.
        """
        self._discrete = discrete
        if len(position) != 2:
            raise ValueError("Observation position must be a vector of length 2")
        if self._discrete:
            self.position = position
        else:
            # self.position = (round(position[0], Observation.PRECISION),
            #                  round(position[1], Observation.PRECISION))
            self.position = (position[0], position[1])

    def discretize(self):
        return Observation(self.position, discrete=True)

    def __hash__(self):
        return hash(self.position)
    
    def __eq__(self, other):
        if isinstance(other, Observation):
            return self.position == other.position
        else:
            return False
        
    def __str__(self):
        return self.__repr__()
    
    def __repr__(self):
        return "Observation(%s)" % (str(self.position))


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
        expected_position = tuple(self.func(state.position, action))
        if next_state.position == expected_position:
            return 1.0 - self._epsilon
        else:
            return self._epsilon

    def sample(self, state, action):
        next_state = copy.deepcopy(state)
        next_state.position = tuple(self.func(state.position, action))
        return next_state

    def argmax(self, state, action):
        """Returns the most likely next state"""
        return self.sample(state, action)

    def func(self, state_pos, action):
        """Returns the function of the underlying system dynamics.
        The function is: (xt, ut) -> xt+1 where xt, ut, xt+1 are
        all numpy arrays."""
        return np.array([state_pos[0] + action[0],
                        state_pos[1] + action[1]])
    
    def func_noise(self, var_sysd=1e-9):
        """Returns a function that returns a state-dependent Gaussian noise."""
        def fn(mt):
            gaussian_noise = Gaussian([0,0],
                                      [[var_sysd, 0],
                                      [0, var_sysd]])
            return gaussian_noise
        return fn


class ObservationModel(ObservationModel):

    def __init__(self, light, const):
        """
        `light` and `const` are parameters in
        :math:`w(x) = \frac{1}{2}(\text{light}-s_x)^2 + \text{const}`

        They should both be floats. The quantity :math:`w(x)` will
        be used as the variance of the covariance matrix in the gaussian
        distribution (this is how I understood the paper).
        """
        self._light = light
        self._const = const

    def _compute_variance(self, pos):
        return 0.5 * (self._light - pos[0])**2 + self._const

    def noise_covariance(self, pos):
        variance = self._compute_variance(pos)
        return np.array([[variance, 0],
                         [0, variance]])

    # |FIXME| observe according to true/belief state??
    def probability(self, observation, next_true_state, next_belief, action):
        """
        The observation is :math:`g(x_t) = x_t+\omega`. So
        the probability of this observation is the probability
        of :math:`\omega` which follows the Gaussian distribution.
        """
        # if self._discrete:
        #     observation = observation.discretize()
        variance = self._compute_variance(next_true_state.position)
        gaussian_noise = Gaussian([0,0],
                                  [[variance, 0],
                                   [0, variance]])
        omega = (observation.position[0] - next_belief.position[0],
                 observation.position[1] - next_belief.position[1])
        return gaussian_noise[omega]

    def sample(self, next_state, action, argmax=False):
        """sample an observation."""
        # Sample a position shift according to the gaussian noise.
        obs_pos = self.func(next_state.position, False)
        return Observation(tuple(obs_pos))
        
    def argmax(self, next_state, action):
        return self.sample(next_state, action, argmax=True)

    def func(self, next_state_pos, mpe=False):
        variance = self._compute_variance(next_state_pos)
        gaussian_noise = Gaussian([0,0],
                                  [[variance, 0],
                                   [0, variance]])
        if mpe:
            omega = gaussian_noise.mpe()
        else:
            omega = gaussian_noise.random()
        return np.array([next_state_pos[0] + omega[0],
                         next_state_pos[1] + omega[1]])

    def func_noise(self):
        """Returns a function that returns a state-dependent Gaussian noise."""
        def fn(mt):
            variance = self._compute_variance(mt)
            gaussian_noise = Gaussian([0,0],
                                      [[variance, 0],
                                       [0, variance]])
            return gaussian_noise
        return fn


class RewardModel(RewardModel):
    def __init__(self, light, goal_state, epsilon=0.25):
        self.light = light
        self._goal_state = goal_state
        self._epsilon=epsilon

    def _reward_func_state(self, state: State, action, next_state: State, goal_state: State, epsilon):
        if np.sum((np.asarray(goal_state.position) - np.asarray(next_state.position))**2) < epsilon**2:
            reward = 100
        else:
            # reward = (-1) * np.abs(next_state.position[0] - self.light)
            reward = -1
        return reward

        # # Euclidean distance
        # reward = (-1)*np.sum((np.asarray(goal_state.position) - np.asarray(next_state.position))**2)
        # return reward

    def _reward_func_particle(self, state: Particles, action: Action, next_state: Particles, goal_state: State, epsilon: float):
        # for belief state - be used to real action
        reward_expectatation = 0
        for state_in in next_state.particles:
            reward_expectatation += self._reward_func_state(state_in, action, state_in, self._goal_state, self._epsilon)
        return reward_expectatation / len(next_state)

    def _reward_func_hist(self, state: Histogram, action: Action, next_state: Histogram, goal_state: State, epsilon: float):
        # for belief state - be used to real action
        normalized_hist = next_state.get_normalized()
        reward_expectatation = 0
        for state_in in normalized_hist:
            reward_expectatation += normalized_hist[state_in] * self._reward_func_state(state_in, action, state_in, self._goal_state, self._epsilon)
        return reward_expectatation

    def sample(self, state, action, next_state):
        # |TODO| make exception
        # deterministic      
        if str(type(next_state)) == "<class '__main__.State'>":
            return self._reward_func_state(state, action, next_state, self._goal_state, self._epsilon)
        # |TODO| currently, get reward after update, can make same as state case(before update)?
        elif str(type(next_state)) == "<class 'POMDP_framework.Particles'>":
            return self._reward_func_particle(state, action, next_state, self._goal_state, self._epsilon)
        elif str(type(next_state)) == "<class 'POMDP_framework.Histogram'>":
            return self._reward_func_hist(state, action, next_state, self._goal_state, self._epsilon)
        
    # For State
    def is_goal_state(self, state: State):
        if np.sum((np.asarray(self._goal_state.position) - np.asarray(state.position))**2) < self._epsilon**2:
            return True
        return False

    # For Histogram
    def is_goal_hist(self, state: Histogram, thres=0.7):
        # test goal condition: #particle(prob) in goal_state >= thres -> True
        prob_in_goal = 0
        normalized_hist = state.get_normalized()
        for particle in normalized_hist:
            if np.sum((np.asarray(self._goal_state.position) - np.asarray(particle.position))**2) < self._epsilon**2:
                prob_in_goal += normalized_hist[particle]
        print(r"% of particles in goal: " + str(prob_in_goal*100) + "%")
        return prob_in_goal >= thres
    
    # For Particles
    def is_goal_particles(self, state: Particles, thres=0.7):
        # test goal condition: #particle(prob) in goal_state >= thres -> True
        num_particles = len(state)
        prob_in_goal = 0
        for particle in state.particles:
            if np.sum((np.asarray(self._goal_state.position) - np.asarray(particle.position))**2) < self._epsilon**2:
                prob_in_goal += 1
        prob_in_goal /= num_particles
        print(r"% of particles in goal: " + str(prob_in_goal*100) + "%")
        return prob_in_goal >= thres


class PolicyModel(RandomRollout):
    """
    This is an extremely dumb policy model; To keep consistent with the framework.
    """
    pass


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
    def __init__(self, init_state, init_belief, goal_state, light, const, guide_policy=None):
        if guide_policy is not None:
            # nn_config = Settings()
            # guide_policy = NNRegressionPolicyModel(nn_config)
            agent = Agent(init_belief,
                      guide_policy,
                      TransitionModel(),
                      ObservationModel(light,const),
                      RewardModel(light, goal_state))
        else:
            agent = Agent(init_belief,
                        PolicyModel(),
                        TransitionModel(),
                        ObservationModel(light,const),
                        RewardModel(light, goal_state))

        env = LightDarkEnvironment(init_state,                  # init state
                                   light,                       # light
                                   const,                       # const
                                   RewardModel(light, goal_state))     # reward model
        
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


class LightDarkViz:
    """This class deals with visualizing a light dark domain"""

    def __init__(self, problem: LightDarkProblem, x_range, y_range, res):
        """
        Args:
            env (LightDarkEnvironment): Environment for light dark domain.
            x_range (tuple): a tuple of floats (x_min, x_max).
            y_range (tuple): a tuple of floats (y_min, y_max).
            res (float): specifies the size of each rectangular strip to draw;
                As in the paper, the light is at a location on the x axis.
        """
        self._env = problem.env
        self._res = res
        self._x_range = x_range
        self._y_range = y_range
        fig = plt.gcf()
        self._ax = fig.add_subplot(1,1,1)
        self._goal_pos = problem.goal_state.position
        self._init_pos = problem.env.init_state.position
        # self._m_0 = None  # initial belief pose
        self._init_belief = []
        for p in problem.agent._init_belief.particles:
            self._init_belief.append(p.position)
        self._init_belief = np.asarray(self._init_belief).T

        # For tracking the path; list of robot position tuples
        self._log_paths = {}
        self._log_state = []
        self._log_belief = []
        self._log_belief_expectation = []

    def log_position(self, position, path=0):
        if path not in self._log_paths:
            self._log_paths[path] = []
        self._log_paths[path].append(position)
    
    def log_state(self, state):
        log = state.position
        log = np.asarray(log).T
        self._log_state.append(log)

    def log_belief_expectation(self, belief_expectation):
        log = np.asarray(belief_expectation).T
        self._log_belief_expectation.append(log)

    def log_belief(self, belief):
        log = []
        for p in belief.particles:
            log.append(p.position)
        log = np.asarray(log).T
        self._log_belief.append(log)

    def set_goal(self, goal_pos):
        self._goal_pos = goal_pos

    def set_init_state(self, init_pos):
        self._init_pos = init_pos

    # def set_initial_belief_pos(self, m_0):
    #     self._m_0 = m_0

    # |FIXME| change Particles to array
    def set_init_belief(self, init_belief):
        self._init_belief = init_belief

    def plot(self,
             path_colors={0: [(0,0,0), (0,0,0)]},
             path_styles={0: "-"},
             path_widths={0: 1}):
        self._plot_gradient()
        self._plot_path(path_colors, path_styles, path_widths)
        # self._plot_robot()
        self._plot_goal()
        self._plot_init_state()
        # self._plot_initial_belief_pos()
        self._plot_initial_belief()
        self._plot_state()
        self._plot_log_belief()
        self._plot_log_belief_expectation()

    def _plot_robot(self):
        cur_pos = self._env.state.position
        util.plot_circle(self._ax, cur_pos,
                         0.25, # tentative
                         color="black", fill=False,
                         linewidth=1, edgecolor="black",
                         zorder=3)

    # def _plot_initial_belief_pos(self):
    #     if self._m_0 is not None:
    #         util.plot_circle(self._ax, self._m_0,
    #                          0.25, # tentative
    #                          color="black", fill=False,
    #                          linewidth=1, edgecolor="black",
    #                          zorder=3)

    def _plot_goal(self):
        if self._goal_pos is not None:
            util.plot_circle(self._ax,
                             self._goal_pos,
                             0.25,  # tentative
                             linewidth=1, edgecolor="blue",
                             zorder=3)
    
    def _plot_init_state(self):
        plt.scatter(self._init_pos[0], self._init_pos[1], c = 'k')
    
    def _plot_state(self):
        s = np.asarray(self._log_state).T
        plt.scatter(s[0], s[1], c='k')
        plt.plot(s[0], s[1], c='k')

    def _plot_initial_belief(self):
        if self._init_belief is not None:
            plt.scatter(self._init_belief[0], self._init_belief[1], s=0.1 ,c='g')
    
    def _plot_log_belief(self):
        for b in self._log_belief:
            plt.scatter(b[0], b[1], s=0.1 ,c='g')

    def _plot_log_belief_expectation(self):
        p = np.asarray(self._log_belief_expectation).T
        plt.scatter(p[0], p[1], c='r')
        plt.plot(p[0], p[1], c='r')


    def _plot_path(self, colors, styles, linewidths):
        """Plot robot path"""
        # Plot line segments
        for path in self._log_paths:
            if path not in colors:
                path_color = [(0,0,0)] * len(self._log_paths[path])
            else:
                if len(colors[path]) == 2:
                    c1, c2 = colors[path]
                    path_color = util.linear_color_gradient(c1, c2,
                                                            len(self._log_paths[path]),
                                                            normalize=True)
                else:
                    path_color = [colors[path]] * len(self._log_paths[path])

            if path not in styles:
                path_style = "--"
            else:
                path_style = styles[path]

            if path not in linewidths:
                path_width = 1
            else:
                path_width = linewidths[path]

            for i in range(1, len(self._log_paths[path])):
                p1 = self._log_paths[path][i-1]
                p2 = self._log_paths[path][i]
                try:
                    util.plot_line(self._ax, p1, p2, color=path_color[i],
                                   linestyle=path_style, zorder=2, linewidth=path_width)
                except Exception:
                    import pdb; pdb.set_trace()

    def _plot_gradient(self):
        """display the light dark domain."""
        xmin, xmax = self._x_range
        ymin, ymax = self._y_range
        # Note that higher brightness has lower brightness value
        hi_brightness = self._env.const
        lo_brightness = max(0.5 * (self._env.light - xmin)**2 + self._env.const,
                            0.5 * (self._env.light - xmax)**2 + self._env.const)
        # Plot a bunch of rectangular strips along the x axis
        # Check out: https://stackoverflow.com/questions/10550477
        x = xmin
        verts = []
        colors = []
        while x < xmax:
            x_next = x + self._res
            verts.append([(x, ymin), (x_next, ymin), (x_next, ymax), (x, ymax)])
            # compute brightness based on equation in the paper
            brightness = 0.5 * (self._env.light - x)**2 + self._env.const
            # map brightness to a grayscale color
            grayscale = int(round(util.remap(brightness, hi_brightness, lo_brightness, 255, 0)))
            grayscale_hex = util.rgb_to_hex((grayscale, grayscale, grayscale))
            colors.append(grayscale_hex)
            x = x_next
        util.plot_polygons(verts, colors, ax=self._ax)
        self._ax.set_xlim(xmin, xmax)
        self._ax.set_ylim(ymin, ymax)


def main():
    plotting = False
    save_log = False
    save_data = False

    guide = True
    rollout_guide = False

    num_sucess = 0
    num_fail = 0
    num_planning = 100
    num_particles = 100
    init_random_range = 0

    if save_data:
        save_dir = os.path.join(os.getcwd(),'result/dataset','long_1K_10')
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

    log_time = []
    log_val_success =[]
    log_val_fail = []

    if guide:
        nn_config = Settings()
        guide_policy = NNRegressionPolicyModel(nn_config)
    else:
        guide_policy = None

    for n in range(num_planning):
        print("========================================================") 
        print("========================= %d-th ========================" % (n+1)) 
        print("========================================================") 
        init_state = State(tuple(2.5 + init_random_range * (np.random.rand(2) - 0.5)))
        goal_state = State((0, 0))

        # inital belief state is uniformly distribution
        # init_belief = Histogram({})
        init_belief = []
        # # For uniform initialization
        # while len(init_belief) < num_particles:
        #     sample = State(tuple(2.5 + random_range * (np.random.rand(2)-0.5)))
        #     init_belief.append(sample)
            # init_belief[sample] = 1 / (random_range**2 * num_particles)
        # For gaussian initalization
        init_belief_std = 0.25
        while len(init_belief) < num_particles:
            sample = State(tuple(np.asarray(init_state.position) + init_belief_std * (np.random.randn(2))))
            init_belief.append(sample)
        init_belief = Particles(init_belief)
        
        # defines the observation noise equation.
        light = 5.0
        const = 0.001

        # planning horizon
        planning_horizon = 30

        # defines discount_factor
        discont_factor = 0.9
        
        # creates POMDP model
        light_dark_problem = LightDarkProblem(init_state, init_belief, goal_state, light, const, guide_policy)
        # light_dark_problem.agent.set_belief(Particles.from_histogram(init_belief,num_particles=1))
        light_dark_problem.agent.set_belief(init_belief)

        # set planner
        planner = POMCPOW(pomdp=light_dark_problem, max_depth=planning_horizon, planning_time=-1., num_sims=num_particles,
                        discount_factor=discont_factor, exploration_const=math.sqrt(2),
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
        total_plan_time = 0.0
        log_time_each = []
        log_val_each = []
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

            best_action, time_taken, sims_count, tree_value = planner.plan(light_dark_problem.agent, i, logging, save_data, guide, rollout_guide)
            
            log_time_each.append(time_taken)
            log_val_each.append(tree_value)
            log_time_each_avg = np.mean(np.asarray(log_time_each))
            log_val_each_avg = np.mean(np.asarray(log_val_each))

            # |FIXME|
            next_state = light_dark_problem.agent.transition_model.sample(light_dark_problem.env.state, best_action)
            real_observation = light_dark_problem.agent.observation_model.sample(next_state, best_action)
            
            # select observataion node in existing node - unrealistic
            # real_observation = random.choice(list(planner._agent.tree[best_action].children.keys()))
            
            total_num_sims += sims_count
            total_plan_time += time_taken

            check_goal = planner.update(light_dark_problem.agent, light_dark_problem.env, best_action, next_state, real_observation)
            # |TODO| can move before update to avoid confusion state case and belief case?
            # By belief state
            reward = light_dark_problem.env.reward_model.sample(light_dark_problem.agent.cur_belief, best_action, light_dark_problem.agent.cur_belief)
            # # By true state
            # reward = light_dark_problem.env.reward_model.sample(next_state, best_action, next_state)

            # |NOTE| only take positive reward as achieving goal condition
            if not check_goal:
                reward = -1

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
                # print("History:", planner.history)
                print("Total Num sims: %d" % total_num_sims)
                print("Total Plan time: %.5f" % total_plan_time)
                num_sucess += 1
         
                log_val_success.append(log_val_each_avg)
                
                # # save data
                # if save_data:
                #     with open(os.path.join(save_dir,'success_history.pickle'), 'ab') as f:
                #         pickle.dump(planner.history[:-1], f, pickle.HIGHEST_PROTOCOL)
                #     with open(os.path.join(save_dir,'success_value.pickle'), 'ab') as f:
                #         pickle.dump(total_reward, f, pickle.HIGHEST_PROTOCOL)
                
                # Saving success history
                if save_data:
                    with open(os.path.join(save_dir, 'simulation_history_data.pickle'), 'ab') as f:
                        pickle.dump(planner.history_data, f)
                
                break

            elif i == planning_horizon-1:
                print("==== Fail ====")
                print("Total reward: %.5f" % total_reward)
                # print("History:", planner.history)
                print("Total Num sims: %d" % total_num_sims)
                print("Total Plan time: %.5f" % total_plan_time)
                num_fail += 1

                log_val_fail.append(log_val_each_avg)

                # # save data
                # if save_data:
                #     with open(os.path.join(save_dir,'fail_history.pickle'), 'ab') as f:
                #         pickle.dump(planner.history[:-1], f, pickle.HIGHEST_PROTOCOL)
                #     with open(os.path.join(save_dir,'fail_value.pickle'), 'ab') as f:
                #         pickle.dump(total_reward, f, pickle.HIGHEST_PROTOCOL)

                # Saving fail history
                if save_data:
                    with open(os.path.join(save_dir, 'simulation_history_data.pickle'), 'ab') as f:
                        pickle.dump(planner.history_data, f)
            
        if plotting:
            viz.plot(path_colors={0: [(0,0,0), (0,255,0)],
                                    1: [(0,0,0), (255,0,0)]},
                        path_styles={0: "--",
                                    1: "-"},
                        path_widths={0: 4,
                                    1: 1})
            plt.show()

        log_time.append(log_time_each_avg)

        print("num_sucess: %d" % num_sucess)
        print("num_fail: %d" % num_fail)

    time_mean = np.mean(np.asarray(log_time))
    time_std = np.std(np.asarray(log_time))
    val_success_mean = np.mean(np.asarray(log_val_success))
    val_success_std = np.std(np.asarray(log_val_success))
    val_fail_mean = np.mean(np.asarray(log_val_fail))
    val_fail_std = np.std(np.asarray(log_val_fail))

    print("====Finish===")
    print("total_num_sucess: %d" % num_sucess)
    print("total_num_fail: %d" % num_fail)
    print("planning time(mean): %f" % time_mean)
    print("planning time(std): %f" % time_std)
    print("value of tree(success/mean): %f" % val_success_mean)
    print("value of tree(success/std): %f" % val_success_std)
    print("value of tree(fail/mean): %f" % val_fail_mean)
    print("value of tree(fail/std): %f" % val_fail_std)
      

if __name__ == '__main__':
    main()