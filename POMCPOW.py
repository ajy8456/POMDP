from POMDP_framework import *
from POMCP import *
# |TODO| How to use light_dark_problem.State
from light_dark_problem import State, expectation_belief
import os
import copy
import time
import random
import math
import pickle
import numpy as np

import pdb


class POMCPOW(Planner):
    def __init__(self, pomdp, 
                 max_depth, planning_time=-1., num_sims=-1,
                 discount_factor=0.9, save_dir_sim=None, exploration_const=math.sqrt(2),
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

        # For logging simulation result
        # dict: {key: str(sims_count), value: Tuple(log_path: list[np.array], total_reward: float)}
        self.log = {}
        self.path = None
        # self.history_data = []
        self.save_dir_sim = save_dir_sim
        self.num_sim_data = 0
        self.sims_count_success = 0


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
    
    def plan(self, agent, horizon, logging=False, save_sim_name=False, randomize=False, guide_policy=False, rollout_guide=False, guide_value=False):
        # Only works if the agent's belief is particles
        if not isinstance(agent.belief, Particles):
            raise TypeError("Agent's belief is not represented in particles.\n"\
                            "POMCP not usable. Please convert it to particles.")

        self._agent = agent   # switch focus on planning for the given agent
        if not hasattr(self._agent, "tree"):
            self._agent.add_attr("tree", None)
        
        sims_count = 0
        time_taken = 0.0
        start_time = time.time()

        while True:
            state = self._agent.sample_belief()
        
            if logging:
                self.log_path = []
                self.log_path.append(state.position)
        

            # ################################################################
            # print(f"=============Start Simulation: {sims_count}=================")
            # print("call _simulate()")
            # ################################################################

            # contain excuted history
            simulate_history = agent.history
            # init_observation = agent._observation_model.sample(state, (0,0))
            # simulate_history = (((0,0), init_observation.position, state.position, 0),)

            total_reward = self._simulate(state, simulate_history, self._agent.tree, None, None, len(simulate_history), logging=logging, save_sim_name=save_sim_name, guide_policy=guide_policy, rollout_guide=rollout_guide, guide_value=guide_value, randomize=randomize)
            # total_reward = self._simulate(state, simulate_history, self._agent.tree, None, None, len(simulate_history), logging=logging, guide=guide, rollout_guide=rollout_guide)


            # ################################################################
            # print(f"=============End Simulation: {sims_count}=================")
            # ################################################################
        
            if logging:
                self.log[str(sims_count)] = (self.log_path, total_reward)
        
            sims_count +=1
            time_taken = time.time() - start_time
            if self._planning_time > 0 and time_taken > self._planning_time:
                break
            if self._num_sims > 0 and sims_count >= self._num_sims:
                break


        best_action = self._agent.tree.argmax()
        self._last_num_sims = sims_count
        self._last_planning_time = time_taken

        # |TODO| how to remove the local variable "num_sims_success"?
        num_sims_success = self.sims_count_success
        self.sims_count_success = 0

        # logging first planning's path, QNode
        if logging:
            with open('simulation_log_.pickle', 'wb') as f:
                pickle.dump(self.log, f)
            with open('simulation_tree_.pickle', 'wb') as f:
                log_value = {}
                for key in self._agent.tree.children.keys():
                    log_value[key] = self._agent.tree.children[key].value
                pickle.dump(log_value, f)

        # collect data using result of search: (history: tuple(a,o,s',r), next_actions: list[tuple, ...], p_action: np.array, num_visits_action: np.array, val_action: np.array, val_root: float) + total_reward(@finishing the problem)
        sampled_action = []
        val_action = []
        num_visits_action = []
        for k in self._agent.tree.children.keys():
            sampled_action.append(k)
            val_action.append(self._agent.tree.children[k].value)
            num_visits_action.append(self._agent.tree.children[k].num_visits)
        val_action = np.asarray(val_action)
        num_visits_action = np.asarray(num_visits_action)
        # |TODO| introduce the temperature factor?
        p_action = num_visits_action/self._agent.tree.num_visits
        val_root = np.dot(val_action, p_action).item()
        self._agent.tree.value = val_root

        data = (agent.history, sampled_action, p_action, num_visits_action, val_action, val_root)

        # next_action: sampling with probabilty of p_action or best_action
        next_action = best_action
        # next_action = sampled_action[np.random.choice(np.arange(len(p_action)), p=p_action)]

        # print("b0:", self._agent.tree.belief.particles)
        return next_action, time_taken, sims_count, num_sims_success, val_root, self._agent.tree[best_action].value, data

    # |NOTE| uniformly random
    # |TODO| move to light_dark_problem.py and make NotImplementedError because this is only for light dark domain
    # def _NextAction(self, state, x_range=(-1,6), y_range=(-1,6)):
    #     pos = state.position
    #     _action_x = random.uniform(x_range[0] - pos[0], x_range[1] - pos[0])
    #     _action_y = random.uniform(y_range[0] - pos[1], y_range[1] - pos[1])
    #     _action = (_action_x,_action_y)
    #     return _action

    def _ActionProgWiden(self, vnode, history, state, guide_policy, k_a=2, alpha_a=1/2):
        _history = vnode

        # # For experiment
        # # if len(history) == 1:
        # #     test_action = (-2.5, -2.5)
        # if len(history) == 1:
        #     test_action = (2.5, -1.25)
        # # elif len(history) == 2:
        # #     test_action = (-5.0, -1.25)
        # else:
        #     test_action = self._NextAction(state)
        
        # if vnode[test_action] is None:
        #     history_action_node = QNode(self._num_visits_init, self._value_init)
        #     vnode[test_action] = history_action_node
        
        # return test_action

        if len(_history.children) <= k_a*_history.num_visits**alpha_a:
            if guide_policy:
                # Adding stochasticity
                if _history.num_visits%4 == 3:
                    _action = self._pomdp.agent._policy_model._NextAction(state)
                else:
                    _action, inference_time = self._agent._policy_model.sample(history, self._pomdp.goal_state.position)
                    # print("Inference time:", inference_time)
            else:
                # # testing for optimal solution
                # if len(history) == 1:
                #     if vnode[(-2.5, -2.5)] is None:
                #         history_action_node = QNode(self._num_visits_init, self._value_init)
                #         vnode[(-2.5, -2.5)] = history_action_node
                #     if vnode[(0.0, 0.0)] is None:
                #         history_action_node = QNode(self._num_visits_init, self._value_init)
                #         vnode[(0.0, 0.0)] = history_action_node
                #     _action = (2.5, -1.25)
                # elif len(history) == 2:
                #     if history[1][0][0] == 2.5:
                #         # _action = (-state.position[0], -state.position[1])
                #         # _action = (-5.0, -1.25)
                #         avg_belief = expectation_belief(self._agent.belief)
                #         _action = (-(avg_belief[0]+2.5), -(avg_belief[1]-1.25))
                #     else:
                #         _action = self._pomdp.agent._policy_model._NextAction(state)
                # else:
                _action = self._pomdp.agent._policy_model._NextAction(state)

            if vnode[_action] is None:
                history_action_node = QNode(self._num_visits_init, self._value_init)
                vnode[_action] = history_action_node
        
        # ################################################################
        # num_q = len(_history.children)
        # print(f"#Q children: {num_q}")
        # print("call _ucb()")
        # ################################################################

        # # testing for optimal solution
        # if len(history) == 1:
        #     a_candidate = [(-2.5,-2.5), (2.5,-1.25), (0.0, 0.0)]
        #     a_idx = np.random.randint(0,3)
        #     return a_candidate[a_idx]

        return self._ucb(vnode)

    def _simulate(self, state, history, root, parent, observation, depth, logging, save_sim_name, guide_policy, rollout_guide, guide_value, randomize, k_o=2, alpha_o=1/2): # root<-class:VNode, parent<-class:QNode
    # def _simulate(self, state, history, root, parent, observation, depth, logging, guide, rollout_guide, k_o=2, alpha_o=1/2): # root<-class:VNode, parent<-class:QNode
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
        
        # ################################################################
        # print(f"depth: {depth}")
        # print("call _ActionProgWiden()")
        # ################################################################

        action = self._ActionProgWiden(vnode=root, history=history, state=state, guide_policy=guide_policy)

        # ################################################################
        # print(f"return _ucb() & _ActionProgWiden(): action {action}")
        # ################################################################

        next_state, observation, reward, nsteps = sample_generative_model(self._agent, state, action)

        # ################################################################
        # print(f"next_state: {next_state}, reward: {reward}")
        # ################################################################

        if logging:
            self.log_path.append(next_state.position)

        if len(root[action].children) <= k_o*root[action].num_visits**alpha_o:
            if root[action][observation] is None:
                # |FIXME| M(hao) <- M(hao) + 1
                pass
        else:
            # |FIXME| o <- selet w.p. M(hao)/SIGMA_o M(hao)
            observation = random.choice(list(root[action].children.keys()))

            # ################################################################
            # num_v = len(root[action].children)
            # print(f"#V children is full: {num_v}")
            # ################################################################

        prob = self._pomdp.agent._observation_model.probability(observation, next_state, next_state, action)

        if observation not in root[action].children:
            # |NOTE| append s` to B(hao)
            # |NOTE| append Z(o|s,a,s`) to W(hao)
            root[action][observation] = self._VNode(agent=self._agent, root=False)
            root[action][observation].belief[next_state] = prob

            history += ((action, observation.position, next_state.position, reward), )

            # |NOTE| if new observation node already achieve goal condition, then return simulate() without calling rollout()
            if self._pomdp.env.reward_model.is_goal_state(next_state):

                total_reward = reward
        
                root.num_visits += 1
                root[action].num_visits += 1
                root[action].value = root[action].value + (total_reward - root[action].value) / (root[action].num_visits)

                # ################################################################
                # print("!!!!!!!Goal!!!!!!")
                # print(f"return _simulate(), don't call _rollout(), total_reward: {total_reward}")
                # ################################################################

                # Saving success history
                self.sims_count_success += 1
                if save_sim_name:
                    # self.history_data.append(history)
                    sim_data = np.asarray(history)
                    if randomize:
                        sim_data = (sim_data, self._pomdp.goal_state.position)
                    with open(os.path.join(self.save_dir_sim, f'{save_sim_name}_{self.num_sim_data}.pickle'), 'wb') as f:
                        pickle.dump(sim_data, f)
                    self.num_sim_data += 1
                    
                return total_reward

            # ################################################################
            # print("new VNode")
            # print("call _rollout()")
            # ################################################################

            if guide_value:
                accumulated_reward_const = self._pomdp.env.reward_model.valuenet.const
                if accumulated_reward_const[0]: # rollout and value network
                    accumulated_reward = accumulated_reward_const[0] * self._rollout(next_state, history, root[action][observation], depth+1, logging=logging, save_sim_name=save_sim_name, rollout_guide=rollout_guide, randomize=randomize) \
                                       + accumulated_reward_const[1] * self._pomdp.env.reward_model.valuenet.sample(history, self._pomdp.goal_state.position)[0]
                else: # value network only
                    accumulated_reward = self._pomdp.env.reward_model.valuenet.sample(history, self._pomdp.goal_state.position)[0]
                total_reward = reward + accumulated_reward
            else:
                total_reward = reward + self._rollout(next_state, history, root[action][observation], depth+1, logging=logging, save_sim_name=save_sim_name, rollout_guide=rollout_guide, randomize=randomize)

        else:
            # |NOTE| append s` to B(hao)
            # |NOTE| append Z(o|s,a,s`) to W(hao)
            root[action][observation].belief[next_state] += prob
            # |NOTE| s` <- select B(hao)[i] w.p W(hao)[i]/sigma(j=1~m) W(hao)[j]
            next_state = root[action][observation].belief.random()

            if logging:
                self.log_path.pop()
                self.log_path.append(next_state.position)

            # |NOTE| r <- R(s,a,s`)
            reward = self._agent.reward_model.sample(state, action, next_state)

            history += ((action, observation.position, next_state.position, reward), )

            # ################################################################
            # print(f"sampling new reward, because observation is change: {reward}")
            # ################################################################

            # |NOTE| check goal condition while simulating 
            if self._pomdp.env.reward_model.is_goal_state(next_state):

                total_reward = reward
        
                root.num_visits += 1
                root[action].num_visits += 1
                root[action].value = root[action].value + (total_reward - root[action].value) / (root[action].num_visits)

                # ################################################################
                # print("!!!!!!!Goal!!!!!!")
                # print(f"return _simulate(), total_reward: {total_reward}")
                # ################################################################

                # Saving success history
                self.sims_count_success += 1
                if save_sim_name:
                    # self.history_data.append(history)
                    sim_data = np.asarray(history)
                    if randomize:
                        sim_data = (sim_data, self._pomdp.goal_state.position)
                    with open(os.path.join(self.save_dir_sim, f'{save_sim_name}_{self.num_sim_data}.pickle'), 'wb') as f:
                        pickle.dump(sim_data, f)
                    self.num_sim_data += 1

                return total_reward

            # ################################################################
            # print(f"call nested _simulate(), depth: {depth}")
            # ################################################################

            total_reward = reward + (self._discount_factor**nsteps)*self._simulate(next_state,
                                                                            history,
                                                                            root[action][observation],
                                                                            root[action],
                                                                            observation,
                                                                            depth+nsteps,
                                                                            logging=logging,
                                                                            save_sim_name=save_sim_name,
                                                                            guide_policy=guide_policy,
                                                                            rollout_guide=rollout_guide,
                                                                            guide_value=guide_value,
                                                                            randomize=randomize)
        
            # ################################################################
            # print("return nested _simulate()")
            # ################################################################

        # |TODO| agent.tree->Q->V check
        root.num_visits += 1
        root[action].num_visits += 1
        root[action].value = root[action].value + (total_reward - root[action].value) / (root[action].num_visits)

        # ################################################################
        # print(f"return _simulate(), total_reward: {total_reward}")
        # ################################################################

        return total_reward

    def _rollout(self, state, history, root, depth, logging, save_sim_name, randomize, rollout_guide=False): # root<-class:VNode
        discount = self._discount_factor
        total_discounted_reward = 0.0

        while depth < self._max_depth:
            if rollout_guide:
                action, _ = self._agent._policy_model.sample(history)
            else:
                action = _action = self._pomdp.agent._policy_model._NextAction(state)
            next_state, observation, reward, nsteps = sample_generative_model(self._agent, state, action)

            if logging:
                self.log_path.append(next_state.position)

            history += ((action, observation.position, next_state.position, reward), )

            # |NOTE| check goal condition while rollout
            if self._pomdp.env.reward_model.is_goal_state(next_state):

                if reward < 0.0:
                    pdb.set_trace()

                depth += nsteps

                total_discounted_reward += reward * discount
                discount *= (self._discount_factor**nsteps)
                state = next_state

                # ################################################################
                # print("!!!!!!!Goal!!!!!!")
                # print(f"return _rollout(), rollout_reward: {total_discounted_reward}")
                # ################################################################

                # Saving success history
                self.sims_count_success += 1
                if save_sim_name:
                    # self.history_data.append(history)
                    sim_data = np.asarray(history)
                    if randomize:
                        sim_data = (sim_data, self._pomdp.goal_state.position)
                    with open(os.path.join(self.save_dir_sim, f'{save_sim_name}_{self.num_sim_data}.pickle'), 'wb') as f:
                        pickle.dump(sim_data, f)
                    self.num_sim_data += 1
                    
                return total_discounted_reward

            else:
                depth += nsteps

            total_discounted_reward += reward * discount
            discount *= (self._discount_factor**nsteps)
            state = next_state

        # ################################################################
        # print(f"return _rollout(), depth: {depth}")
        # print(f"rollout_reward: {total_discounted_reward}")
        # ################################################################

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
            # if agent is None:
            #     return VNodeParticles(self._num_visits_init,
            #                           self._value_init,
            #                           belief=Histogram({}))
            #                         #   belief=Particles([]))

            # else:
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
        

        # |TODO| From now, move to light_dark_problem.py / how to combine?
        # agent.update_history(real_action, real_observation.position, next_state, real_reward)

        # |FIXME| create new root node
        if agent.tree[real_action][real_observation] is None:
            # # Never anticipated the real_observation. No reinvigoration can happen.
            # raise ValueError("Particle deprivation.")
            agent.tree[real_action][real_observation] = self._VNode(agent=agent, root=True)

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
        # if isinstance(tree_belief, Particles):
        #     agent.set_belief(particle_reinvigoration(tree_belief,
        #                                             len(agent.init_belief),
        #                                             state_transform_func=state_transform_func))

        # |NOTE| bootstrap filtering(when belief is represented by Particels)
        if isinstance(tree_belief, Particles):
            # time_start = time.time()
            new_belief, prediction = bootstrap_filter(tree_belief, next_state, real_action, real_observation, agent.observation_model, agent.transition_model, len(agent.init_belief))
            # time_end = time.time()
            # print("Bootstrap filtering time:", time_end - time_start)
            # check_goal = env.reward_model.is_goal_particles(prediction)
            check_goal = env.reward_model.is_goal_state(next_state)
            agent.set_belief(new_belief)

        # |NOTE| belief state update by Bayes' law(when belief is represented by Histogram)
        if isinstance(tree_belief, Histogram):
            new_belief = update_histogram_belief(tree_belief,
                                                 real_action, real_observation,
                                                 agent.observation_model,
                                                 agent.transition_model)
            agent.set_belief(new_belief)        
            # |TODO| change as Particles - check using prediction
            # check_goal = env.reward_model.is_goal_hist(new_belief)
            check_goal = env.reward_model.is_goal_state(next_state)

        if agent.tree is not None:
            agent.tree.belief = copy.deepcopy(agent.belief)
        
        return check_goal