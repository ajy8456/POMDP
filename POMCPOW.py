from POMDP_framework import Particles
from POMDP_framework import *
from POMCP import *
# |TODO| How to use light_dark_problem.State
from light_dark_problem import State
import copy
import time
import random
import math
import numpy as np


class POMCPOW(Planner):
    def __init__(self, pomdp,
                 max_depth=5, planning_time=-1., num_sims=-1,
                 discount_factor=0.9, exploration_const=math.sqrt(2),
                 num_visits_init=0, value_init=0,
                 action_prior=None):
        """
        Args:
            max_depth (int): Depth of the MCTS tree. Default: 5.
            planning_time (float), amount of time given to each planning step (seconds). Default: -1.
                if negative, then planning terminates when number of simulations `num_sims` reached.
                If both `num_sims` and `planning_time` are negative, then the planner will run for 1 second.
            num_sims (int): Number of simulations for each planning step.
                If negative, then will terminate when planning_time is reached.
                If both `num_sims` and `planning_time` are negative, then the planner will run for 1 second.
            rollout_policy (RolloutPolicy): rollout policy. Default: RandomRollout.
            action_prior (ActionPrior|TODO|): a prior over preferred actions given state and history.
        """
        self._pomdp = pomdp
        self._max_depth = max_depth
        self._planning_time = planning_time
        self._num_sims = num_sims
        if self._num_sims < 0 and self._planning_time < 0:
            self._planning_time = 1.
        self._num_visits_init = num_visits_init
        self._value_init = value_init
        self._discount_factor = discount_factor
        self._exploration_const = exploration_const
        self._action_prior = action_prior
        # to simplify function calls; plan only for one agent at a time
        self._agent = None
        self._last_num_sims = -1
        self._last_planning_time = -1

        # self.history = tuple(pomdp.agent._init_belief)
        # self.history.append(list(list(pomdp.agent._init_belief.histogram.keys())[0].position))
        self.history = []


    @property
    def update_agent_belief(self):
        """True if planner's update function also updates agent's belief."""
        return True

    @property
    def last_num_sims(self):
        """Returns the number of simulations ran for the last `plan` call."""
        return self._last_num_sims

    @property
    def last_planning_time(self):
        """Returns the amount of time (seconds) ran for the last `plan` call."""
        return self._last_planning_time
    
    def plan(self, agent, horizon):
        # # Only works if the agent's belief is particles
        # if not isinstance(agent.belief, Particles):
        #     raise TypeError("Agent's belief is not represented in particles.\n"\
        #                     "POMCP not usable. Please convert it to particles.")

        self._agent = agent   # switch focus on planning for the given agent
        if not hasattr(self._agent, "tree"):
            self._agent.add_attr("tree", None)
        
        sims_count = 0
        time_taken = 0.0
        start_time = time.time()

        while True:
            # # initial belief state is gaussian
            # if horizon == 0:
            #     _init_belief = State(tuple(np.array([self._pomdp.env.init_state.position[0]+init_belief_variance*np.random.randn(), self._pomdp.env.init_state.position[1]+init_belief_variance*np.random.randn()])))
            #     gaussian_noise = Gaussian([0,0],
            #                               [[init_belief_variance, 0],
            #                                [0, init_belief_variance]])
            #     omega = (_init_belief.position[0] - self._pomdp.env.init_state.position[0],
            #              _init_belief.position[1] - self._pomdp.env.init_state.position[1])
            #     prob = gaussian_noise[omega]
            #     self._agent._cur_belief[_init_belief] = prob 
            #     state = _init_belief
            # else:
            #     state = self._agent.sample_belief()
            state = self._agent.sample_belief()
            self._simulate(state, self._agent.history, self._agent.tree, None, None, 0)
            sims_count +=1
            time_taken = time.time() - start_time
            if self._planning_time > 0 and time_taken > self._planning_time:
                break
            if self._num_sims > 0 and sims_count >= self._num_sims:
                break
        if horizon == 0:
            self.history.append(self._agent._cur_belief.get_histogram())
        best_action = self._agent.tree.argmax()
        self._last_num_sims = sims_count
        self._last_planning_time = time_taken
        self._agent.tree.value = self._agent.tree[best_action].value
        return best_action, time_taken, sims_count

    # |NOTE| uniformly random    
    def _NextAction(self):
        _action_x = random.uniform(-1,1)
        _action_y = random.uniform(-1,1)
        _action = (_action_x,_action_y)
        return _action

    def _ActionProgWiden(self, vnode, history, k_a=10, alpha_a=1/10):
        _history = vnode
        if len(_history.children) <= k_a*_history.num_visits**alpha_a:
            _action = self._NextAction()
            if vnode[_action] is None:
                history_action_node = QNode(self._num_visits_init, self._value_init)
                vnode[_action] = history_action_node
        return self._ucb(vnode)

    def _simulate(self, state, history, root, parent, observation, depth, k_o=3, alpha_o=1/3): # root<-class:VNode, parent<-class:QNode
        if depth > self._max_depth:
            return 0
        
        if root is None:
            if self._agent.tree is None:
                root = self._VNode(agent=self._agent, root=True)
                self._agent.tree = root
                if self._agent.tree.history != self._agent.history:
                    raise ValueError("Unable to plan for the given history.")
            else:
                root = self._VNode()

            if parent is not None:
                parent[observation] = root

        action = self._ActionProgWiden(vnode=root, history=history)
        next_state, observation, reward, nsteps = sample_generative_model(self._agent, state, action)
        
        if len(root[action].children) <= k_o*root[action].num_visits**alpha_o:
            if root[action][observation] is None:
                # |FIXME| M(hao) <- M(hao) + 1
                pass
        else:
            # |FIXME| o <- selet w.p. M(hao)/SIGMA_o M(hao)
            observation = random.choice(list(root[action].children.keys()))

        history += ((action, observation), )

        prob = self._pomdp.agent._observation_model.probability(observation, next_state, action)

        if observation not in root[action].children:
            # |NOTE| append s` to B(hao)
            # |NOTE| append Z(o|s,a,s`) to W(hao)
            root[action][observation] = self._VNode(agent=self._agent, root=False)
            root[action][observation].belief[next_state] = prob
            total_reward = reward + self._rollout(next_state, history, root[action][observation], depth+1)
        else:
            # |NOTE| append s` to B(hao)
            # |NOTE| append Z(o|s,a,s`) to W(hao)
            root[action][observation].belief[next_state] += prob
            # |NOTE| s` <- select B(hao)[i] w.p W(hao)[i]/sigma(j=1~m) W(hao)[j]
            next_state = root[action][observation].belief.random()
            # |NOTE| r <- R(s,a,s`)
            reward = self._agent.reward_model.sample(state, action, next_state)
            total_reward = reward + (self._discount_factor**nsteps)*self._simulate(next_state,
                                                                            history,
                                                                            root[action][observation],
                                                                            root[action],
                                                                            observation,
                                                                            depth+nsteps)
        
        # |TODO| agent.tree->Q->V check
        root.num_visits += 1
        root[action].num_visits += 1
        root[action].value = root[action].value + (total_reward - root[action].value) / (root[action].num_visits)
        return total_reward

    def _rollout(self, state, history, root, depth): # root<-class:VNode
        discount = 1.0
        total_discounted_reward = 0.0

        while depth < self._max_depth:
            action = self._NextAction()
            next_state, observation, reward, nsteps = sample_generative_model(self._agent, state, action)
            history = history + ((action, observation),)
            depth += nsteps
            total_discounted_reward += reward * discount
            discount *= (self._discount_factor**nsteps)
            state = next_state
        return total_discounted_reward

    def _ucb(self, root): # rood<-class:VNode
        """UCB1"""
        best_action, best_value = None, float('-inf')
        for action in root.children:
            if root[action].num_visits == 0:
                val = float('inf')
            else:
                val = root[action].value + \
                    self._exploration_const * math.sqrt(math.log(root.num_visits + 1) / root[action].num_visits)
            if val > best_value:
                best_action = action
                best_value = val
        return best_action

    def _sample_generative_model(self, state, action):
        '''
        (s', o, r) ~ G(s, a)
        '''
        if self._agent.transition_model is None:
            next_state, observation, reward = self._agent.generative_model.sample(state, action)
        else:
            next_state = self._agent.transition_model.sample(state, action)
            observation = self._agent.observation_model.sample(next_state, action)
            reward = self._agent.reward_model.sample(state, action, next_state)
        return next_state, observation, reward

    def _VNode(self, agent=None, root=False, **kwargs):
        """
        Returns a VNode with default values;
        The function naming makes it clear that this function is about creating a VNode object.
        """
        if root:
            # agent cannot be None.
            return RootVNodeParticles(self._num_visits_init,
                                      self._value_init,
                                      agent.history,
                                      belief=copy.deepcopy(agent.belief))
        else:
            if agent is None:
                return VNodeParticles(self._num_visits_init,
                                      self._value_init,
                                      belief=Histogram({}))
                                    #   belief=Particles([]))

            else:
                return VNodeParticles(self._num_visits_init,
                                      self._value_init,
                                      belief=Histogram({}))
                                    #   belief=copy.deepcopy(agent.belief))
                                    #   belief=Particles([]))

    def update(self, agent, env, real_action, next_state, real_observation, state_transform_func=None):
        """
        Assume that the agent's history has been updated after taking real_action and receiving real_observation.

        `state_transform_func`: Used to add artificial transform to states during
            particle reinvigoration. Signature: s -> s_transformed
        """
        # if not isinstance(agent.belief, Particles):
        #     raise TypeError("agent's belief is not represented in particles.\n"\
        #                     "POMCP not usable. Please convert it to particles.")
        if not hasattr(agent, "tree"):
            print("Warning: agent does not have tree. Have you planned yet?")
            return
        
        # |FIXME| create new root node
        if agent.tree[real_action][real_observation] is None:
            # # Never anticipated the real_observation. No reinvigoration can happen.
            # raise ValueError("Particle deprivation.")
            agent.tree[real_action][real_observation] = self._VNode(agent=agent, root=False)

        # Update history
        self.history.append(list(real_action))
        self.history.append(list(real_observation.position))

        # Update the state
        env.apply_transition(next_state)

        # Update the tree; Reinvigorate the tree's belief and use it as the updated belief for the agent.
        # use particles of h node
        tree_belief = agent.belief
        agent.tree = RootVNodeParticles.from_vnode(agent.tree[real_action][real_observation], agent.history)
        # # use particles of hao node
        # tree_belief = agent.tree.belief

        # |NOTE| particle reinvigoration(when belief is represented by Particles)
        # If observation was encountered less in simulation, then tree will be None;
        # particle reinvigoration will occur.
        if isinstance(tree_belief, Particles):
            agent.set_belief(particle_reinvigoration(tree_belief,
                                                    len(agent.init_belief),
                                                    state_transform_func=state_transform_func))
            
        # |NOTE| belief state update by Bayes' law(when belief is represented by Histogram)
        if isinstance(tree_belief, Histogram):
            new_belief = update_histogram_belief(tree_belief,
                                                 real_action, real_observation,
                                                 agent.observation_model,
                                                 agent.transition_model)
            agent.set_belief(new_belief)        

        if agent.tree is not None:
            agent.tree.belief = copy.deepcopy(agent.belief)