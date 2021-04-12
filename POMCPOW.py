from POMDP_framework import Action, Agent, POMDP, State, Observation,\
     ObservationModel, TransitionModel, GenerativeDistribution, PolicyModel, Planner,\
     Particles, particle_reinvigoration, sample_generative_model
from POMCP import TreeNode, QNode, VNode, RootVNode, VNodeParticles, RootVNodeParticles,\
    RolloutPolicy, RandomRollout
import copy
import time
import random
import math


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
        # |TODO|
        # self.history

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
    
    def plan(self, agent):
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
            self._simulate(state, self._agent.history, self._agent.tree, None, None, 0)
            sims_count +=1
            time_taken = time.time() - start_time
            if self._planning_time > 0 and time_taken > self._planning_time:
                break
            if self._num_sims > 0 and sims_count >= self._num_sims:
                break
        
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

    def _ActionProgWiden(self, vnode, history, k_a=30, alpha_a=1/30):
        _history = vnode
        if len(_history.children) <= k_a*_history.num_visits**alpha_a:
            _action = self._NextAction()
            if vnode[_action] is None:
                history_action_node = QNode(self._num_visits_init, self._value_init)
                vnode[_action] = history_action_node
                return self._ucb(vnode)

    def _simulate(self, state, history, root, parent, observation, depth, k_o=5, alpha_o=1/15): # root<-class:VNode, parent<-class:QNode
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
        
        _history_action = root[action]
        if len(_history_action.children) <= k_o*_history_action.num_visits**alpha_o:
            if root[action][observation] is None:
                # |TODO| M(hao) <- M(hao)+1 필요 없음?
                # |NOTE| history_action_observation_node : temporal
                _history_action_observation_node = self._VNode(agent=self._agent, root=False)
        else:
            observation = random.choice(root[action].children)
            _history_action_observation_node = root[action][observation]

        history += ((action, observation), )
        # |NOTE| append s` to B(hao)
        # |FIXME| B(hao)에 s`을 추가해야 하는데 B(h)를 복사한 후에 s`을 추가하고 있음
        _history_action_observation_node.belief.add(next_state)
        # |NOTE| append Z(o|s,a,s`) to W(hao)
        prob = self._pomdp.agent._observation_model.probability(observation, next_state, action)
        _history_action_observation_node.belief[next_state] = prob

        if observation not in root[action].children:
            root[action][observation] = _history_action_observation_node
            total_reward = reward + self._rollout(next_state, history, root[action][observation], depth+1)
        else:
            # |NOTE| s` <- select B(hao)[i] w.p W(hao)[i]/sigma(j=1~m) W(hao)[j]
            next_state = root[action][observation].belief.random()
            # |NOTE| r <- R(s,a,s`)
            reward = self._agent.reward_model.sample(state, action, next_state)
            total_reward = reward + (self._discount_factor**nsteps)*self._simulate(next_state,
                                                                            history + ((action, observation),),
                                                                            root[action][observation],
                                                                            root[action],
                                                                            observation,
                                                                            depth+nsteps)
        
        # |FIXME| agent.tree->Q->V check
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
                                      belief=Particles([]))
            else:
                return VNodeParticles(self._num_visits_init,
                                      self._value_init,
                                      belief=copy.deepcopy(agent.belief))

    def update(self, agent, env, real_action, next_state, real_observation, state_transform_func=None):
        """
        Assume that the agent's history has been updated after taking real_action and receiving real_observation.

        `state_transform_func`: Used to add artificial transform to states during
            particle reinvigoration. Signature: s -> s_transformed
        """
        if not isinstance(agent.belief, Particles):
            raise TypeError("agent's belief is not represented in particles.\n"\
                            "POMCP not usable. Please convert it to particles.")
        if not hasattr(agent, "tree"):
            print("Warning: agent does not have tree. Have you planned yet?")
            return
        
        # |FIXME|
        if agent.tree[real_action][real_observation] is None:
            # Never anticipated the real_observation. No reinvigoration can happen.
            raise ValueError("Particle deprivation.")

        # Update the state
        env.apply_transition(next_state)

        # Update the tree; Reinvigorate the tree's belief and use it as the updated belief for the agent.
        agent.tree = RootVNodeParticles.from_vnode(agent.tree[real_action][real_observation], agent.history)
        tree_belief = agent.tree.belief
        agent.set_belief(particle_reinvigoration(tree_belief,
                                                 len(agent.init_belief),
                                                 state_transform_func=state_transform_func))
        # If observation was never encountered in simulation, then tree will be None;
        # particle reinvigoration will occur.
        if agent.tree is not None:
            agent.tree.belief = copy.deepcopy(agent.belief)