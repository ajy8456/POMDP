from POMDP_framework import Action, Agent, POMDP, State, Observation,\
     ObservationModel, TransitionModel, GenerativeDistribution, PolicyModel, Planner,\
     Particles, particle_reinvigoration, sample_generative_model
import copy
import time
import random
import math


class TreeNode:
    def __init__(self):
        self.children = {}
    def __getitem__(self, key):
        if key not in self.children and type(key) == int:
            clist = list(self.children)
            if key >= 0 and key < len(clist):
                return self.children[clist[key]]
            else:
                return None
        return self.children.get(key,None)
    def __setitem__(self, key, value):
        self.children[key] = value
    def __contains__(self, key):
        return key in self.children

class QNode(TreeNode):
    def __init__(self, num_visits, value):
        """
        `history_action`: a tuple ((a,o),(a,o),...(a,)). This is only used for computing hashses
        """
        self.num_visits = num_visits
        self.value = value
        self.children = {}  # o -> VNode
    def __str__(self):
        return "QNode(%.3f, %.3f | %s)" % (self.num_visits, self.value, str(self.children.keys()))
    def __repr__(self):
        return self.__str__()

class VNode(TreeNode):
    def __init__(self, num_visits, value, **kwargs):
        self.num_visits = num_visits
        self.value = value
        self.children = {}  # a -> QNode
    def __str__(self):
        return "VNode(%.3f, %.3f | %s)" % (self.num_visits, self.value, str(self.children.keys()))
    def __repr__(self):
        return self.__str__()
    def print_children_value(self):
        for action in self.children:
            print("   action %s: %.3f" % (str(action), self[action].value))
    def argmax(self):
        """
        Returns the action of the child with highest value
        """
        best_value = float("-inf")
        best_action = None
        for action in self.children:
            if self[action].value > best_value:
                best_action = action
                best_value = self[action].value
        return best_action

class RootVNode(VNode):
    def __init__(self, num_visits, value, history):
        VNode.__init__(self, num_visits, value)
        self.history = history
    @classmethod
    def from_vnode(cls, vnode, history):
        """from_vnode(cls, vnode, history)"""
        rootnode = RootVNode(vnode.num_visits, vnode.value, history)
        rootnode.children = vnode.children
        return rootnode

class VNodeParticles(VNode):
    """POMCP's VNode maintains particle belief"""
    def __init__(self, num_visits, value, belief=Particles([])):
        self.num_visits = num_visits
        self.value = value
        self.belief = belief
        self.children = {}  # a -> QNode
    def __str__(self):
        return "VNode(%.3f, %.3f, %d | %s)" % (self.num_visits, self.value, len(self.belief), str(self.children.keys()))
    def __repr__(self):
        return self.__str__()

class RootVNodeParticles(RootVNode):
    def __init__(self, num_visits, value, history, belief=Particles([])):
        # vnodeobj = VNodeParticles(num_visits, value, belief=belief)
        RootVNode.__init__(self, num_visits, value, history)
        self.belief = belief
    @classmethod
    def from_vnode(cls, vnode, history):
        rootnode = RootVNodeParticles(vnode.num_visits, vnode.value, history, belief=vnode.belief)
        rootnode.children = vnode.children
        return rootnode


class RolloutPolicy(PolicyModel): # for custom rollout policy
    def rollout(self, state, history):
        """rollout(self, State state, tuple history=None)"""
        pass

class RandomRollout(RolloutPolicy):
    def rollout(self, state, history):
        """rollout(self, State state, tuple history=None)"""
        return random.sample(self.get_all_actions(state=state, history=history), 1)[0]


class POUCT(Planner):
    
    """ POUCT (Partially Observable UCT) :cite:`silver2010monte` is presented in the POMCP
    paper as an extension of the UCT algorithm to partially-observable domains
    that combines MCTS and UCB1 for action selection.

    POUCT only works for problems with action space that can be enumerated."""

    def __init__(self,
                 max_depth=5, planning_time=-1., num_sims=-1,
                 discount_factor=0.9, exploration_const=math.sqrt(2),
                 num_visits_init=0, value_init=0,
                 rollout_policy=RandomRollout(),
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
        self._max_depth = max_depth
        self._planning_time = planning_time
        self._num_sims = num_sims
        if self._num_sims < 0 and self._planning_time < 0:
            self._planning_time = 1.
        self._num_visits_init = num_visits_init
        self._value_init = value_init
        self._rollout_policy = rollout_policy
        self._discount_factor = discount_factor
        self._exploration_const = exploration_const
        self._action_prior = action_prior

        # to simplify function calls; plan only for one agent at a time
        self._agent = None
        self._last_num_sims = -1
        self._last_planning_time = -1

    @property
    def updates_agent_belief(self):
        return False

    @property
    def last_num_sims(self):
        """Returns the number of simulations ran for the last `plan` call."""
        return self._last_num_sims

    @property
    def last_planning_time(self):
        """Returns the amount of time (seconds) ran for the last `plan` call."""
        return self._last_planning_time

    def plan(self, agent):
        self._agent = agent   # switch focus on planning for the given agent
        if not hasattr(self._agent, "tree"):
            self._agent.add_attr("tree", None)
        action, time_taken, sims_count = self._search()
        self._last_num_sims = sims_count
        self._last_planning_time = time_taken
        return action

    def update(self, agent, real_action, real_observation):
        """
        update(self, Agent agent, Action real_action, Observation real_observation)
        Assume that the agent's history has been updated after taking real_action and receiving real_observation.
        """
        if not hasattr(agent, "tree") or agent.tree is None:
            print("Warning: agent does not have tree. Have you planned yet?")
            return

        if real_action not in agent.tree or real_observation not in agent.tree[real_action]:
            agent.tree = None  # replan, if real action or observation differs from all branches
        elif agent.tree[real_action][real_observation] is not None:
            # Update the tree (prune)
            agent.tree = RootVNode.from_vnode(agent.tree[real_action][real_observation], agent.history)
        else:
            raise ValueError("Unexpected state; child should not be None")

    def clear_agent(self):
        self._agent = None  # forget about current agent so that can plan for another agent.
        self._last_num_sims = -1

    def _expand_vnode(self, vnode, history, state=None):
        for action in self._agent.valid_actions(state=state, history=history):
            if vnode[action] is None:
                history_action_node = QNode(self._num_visits_init, self._value_init)
                vnode[action] = history_action_node

        if self._action_prior is not None:
            # Using action prior; special values are set;
            for preference in \
                self._action_prior.get_preferred_actions(state, history):
                action, num_visits_init, value_init = preference
                history_action_node = QNode(num_visits_init,
                                            value_init)
                vnode[action] = history_action_node

    def _search(self):
        sims_count = 0
        time_taken = 0.0
        start_time = time.time()
        
        while True:
            ## Note: the tree node with () history will have the init belief given to the agent.
            state = self._agent.sample_belief()
            self._simulate(state, self._agent.history, self._agent.tree, None, None, 0)
            sims_count +=1
            time_taken = time.time() - start_time
            if self._planning_time > 0 and time_taken > self._planning_time:
                break
            if self._num_sims > 0 and sims_count >= self._num_sims:
                break

        best_action = self._agent.tree.argmax()
        self._agent.tree.value = self._agent.tree[best_action].value
        return best_action, time_taken, sims_count

    def _simulate(self, state, history, root, parent, observation, depth): # root<-class:VNode, parent<-class:QNode
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

            self._expand_vnode(root, history, state=state)
            rollout_reward = self._rollout(state, history, root, depth)
            return rollout_reward

        action = self._ucb(root)
        next_state, observation, reward, nsteps = sample_generative_model(self._agent, state, action)
        if nsteps == 0:
            # This indicates the provided action didn't lead to transition
            # Perhaps the action is not allowed to be performed for the given state
            # (for example, the state is not in the initiation set of the option,
            # or the state is a terminal state)
            return reward

        total_reward = reward + (self._discount_factor**nsteps)*self._simulate(next_state,
                                                                               history + ((action, observation),),
                                                                               root[action][observation],
                                                                               root[action],
                                                                               observation,
                                                                               depth+nsteps)
        root.num_visits += 1
        root[action].num_visits += 1
        root[action].value = root[action].value + (total_reward - root[action].value) / (root[action].num_visits)
        return total_reward

    def _rollout(self, state, history, root, depth): # root<-class:VNode
        discount = 1.0
        total_discounted_reward = 0.0

        while depth < self._max_depth:
            action = self._rollout_policy.rollout(state, history)
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
            return RootVNode(self._num_visits_init, float("-inf"), self._agent.history)
        else:
            return VNode(self._num_visits_init, float("-inf"))


class POMCP(POUCT):
    """
    This POMCP version only works for problems with action space that can be enumerated.
    """
    def __init__(self,
                 max_depth=5, planning_time=-1., num_sims=-1,
                 discount_factor=0.9, exploration_const=math.sqrt(2),
                 num_visits_init=0, value_init=0,
                 rollout_policy=RandomRollout(),
                 action_prior=None):
        """
        rollout_policy(vnode, state=?) -> a; default random rollout.
        action_prior (ActionPrior), see above.
        """
        super().__init__(max_depth=max_depth,
                         planning_time=planning_time,
                         num_sims=num_sims,
                         discount_factor=discount_factor,
                         exploration_const=exploration_const,
                         num_visits_init=num_visits_init,
                         value_init=value_init,
                         rollout_policy=rollout_policy,
                         action_prior=action_prior)

    @property
    def update_agent_belief(self):
        """True if planner's update function also updates agent's belief."""
        return True

    def plan(self, agent):
        # Only works if the agent's belief is particles
        if not isinstance(agent.belief, Particles):
            raise TypeError("Agent's belief is not represented in particles.\n"\
                            "POMCP not usable. Please convert it to particles.")
        return POUCT.plan(self, agent)

    def update(self, agent, real_action, real_observation, state_transform_func=None): # ?
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
        
        if agent.tree[real_action][real_observation] is None:
            # Never anticipated the real_observation. No reinvigoration can happen.
            raise ValueError("Particle deprivation.")

        # Update the tree; Reinvigorate the tree's belief and use it as the updated belief for the agent.
        agent.tree = RootVNodeParticles.from_vnode(agent.tree[real_action][real_observation], agent.history)
        tree_belief = agent.tree.belief
        agent.set_belief(particle_reinvigoration(tree_belief,
                                                 len(agent.init_belief.particles),
                                                 state_transform_func=state_transform_func))
        # If observation was never encountered in simulation, then tree will be None;
        # particle reinvigoration will occur.
        if agent.tree is not None:
            agent.tree.belief = copy.deepcopy(agent.belief)

    def _simulate(self, state, history, root, parent, observation, depth):    
        total_reward = POUCT._simulate(self, state, history, root, parent, observation, depth)
        if depth == 1 and root is not None:
            root.belief.add(state)  # belief update happens as simulation goes.
        return total_reward

    def _VNode(self, agent=None, root=False, **kwargs):
        """Returns a VNode with default values; The function naming makes it clear
        that this function is about creating a VNode object."""
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