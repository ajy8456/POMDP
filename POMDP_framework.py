""" For POMDP problem, this module contain basic classes & functions.
class Distrbution/GenerativeDistribution(Distribution)/
class POMDP/State/Action/Observation/TransitionModel/ObservationModel/RewardModel/BlackboxModel/PolicyModel/Agent/Environment
class Option
func sample_explict_models/sample_generative_model
class Particles
func abstraction_over_particles/particle_reinvigoration/update_particles_belief
class Planner
"""

import random
import copy
import sys
import numpy as np

# Check if scipy exists
import importlib
scipy_spec = importlib.util.find_spec("scipy")
if scipy_spec is not None:
    from scipy.linalg import sqrtm
    from scipy.stats import multivariate_normal
else:
    raise ImportError("scipy not found."\
                      "Requires scipy.stats.multivariate_normal to use Gaussian")


class Distribution:
    """
    A Distribution is a probability function that maps from variable value to a real value,
    that is, `Pr(X=x)`.
    """
    def __getitem__(self, varval):
        """
        Probability evaulation.
        Returns the probability of `Pr(X=varval)`.
        """
        raise NotImplementedError
    def __setitem__(self, varval, value):
        """
        __setitem__(self, varval, value)
        Sets the probability of `X=varval` to be `value`.
        """
        raise NotImplementedError
    def __hash__(self):
        return id(self)
    def __eq__(self, other):
        return id(self) == id(other)
    def __iter__(self):
        """Initialization of iterator over the values in this distribution"""
        raise NotImplementedError
    def __next__(self):
        """Returns the next value of the iterator"""
        raise NotImplementedError

    def probability(self, varval):
        return self[varval]

class GenerativeDistribution(Distribution):
    """
    A GenerativeDistribution is a Distribution that additionally exhibits generative properties.
    That is, it supports `argmax(mpe)` and `random` functions.
    """
    def argmax(self, **kwargs):
        """
        synonym for mpe
        """
        return self.mpe(**kwargs)
    def mpe(self, **kwargs):
        """
        Returns the value of the variable that has the highest probability.
        """
        raise NotImplementedError

    def random(self, **kwargs):
        """
        Sample a state based on the underlying belief distribution
        """
        raise NotImplementedError

    def get_histogram(self):
        """
        Returns a dictionary from state to probability
        """
        raise NotImplementedError


class POMDP:
    """
    A POMDP instance <- agent(class:'Agent'), env(class:'Enviroment')

    Because T, R, O may be different for the agent versus the environment, it does not make much sense to have the POMDP class to hold this information.
    Instead, Agent should have its own T, R, O, pi and the Environment should have its own T, R.
    The job of a POMDP is only to verify whether a given state, action, or observation are valid.
    """
    def __init__(self, agent, env, name="POMDP"):
        self.agent = agent
        self.env = env


class State:
    """
    The State class. State must be `hashable`.
    """
    def __eq__(self, other):
        raise NotImplementedError
    def __hash__(self):
        raise NotImplementedError


class Action:
    """
    The Action class. Action must be `hashable`.
    """
    def __eq__(self, other):
        raise NotImplementedError
    def __hash__(self):
        raise NotImplementedError


class Observation:
    """
    The Observation class. Observation must be `hashable`.
    """
    def __eq__(self, other):
        raise NotImplementedError
    def __hash__(self):
        raise NotImplementedError


class TransitionModel:
    """
    T(s,a,s') = Pr(s'|s,a)
    """
    def probability(self, next_state, state, action, **kwargs):
        """
        Args:
            next_state(s') <- basics.State
            state(s) <- basics.State
            action(a) <- basics.action
        Returns:
            Pr(s'|s,a)
        """
        raise NotImplementedError

    def sample(self, state, action, **kwargs):
        """
        Returns next state randomly sampled according to the distribution of this transition model.
        Args:
            state(s) <- basics.State
            action(a) <- basics.action
        Returns:
            next_state(s') <- basics.State, basics.TransitionModel.probability
        """
        raise NotImplementedError
    
    def argmax(self, state, action, **kwargs):
        """
        Returns the most likely next state.
        """
        raise NotImplementedError

    def get_distribution(self, state, action, **kwargs):
        """
        Returns the underlying distribution of the model.
        """
        raise NotImplementedError

    def get_all_states(self):
        """
        Returns a set of all possible states, if feasible.
        """
        raise NotImplementedError


class ObservationModel:
    """
    O(s',a,o) = Pr(o|s',a).
    """
    def probability(self, observation, next_state, action, **kwargs):
        """
        Args:
            observation(o) <- basics.Observation
            next_state(s') <- basics.State
            action(a) <- basics.action
        Returns:
            Pr(o|s',a)
        """
        raise NotImplementedError
    def sample(self, next_state, action, **kwargs):
        """sample(self, next_state, action, **kwargs)
        Returns observation randomly sampled according to the
        distribution of this observation model.

        Args:
            next_state (~pomdp_py.framework.basics.State): the next state :math:`s'`
            action (~pomdp_py.framework.basics.Action): the action :math:`a`
        Returns:
            Observation: the observation :math:`o`
        """
        raise NotImplementedError

    def argmax(self, next_state, action, **kwargs):
        """
        Returns the most likely observation
        """
        raise NotImplementedError

    def get_distribution(self, next_state, action, **kwargs):
        """
        Returns the underlying distribution of the model
        """
        raise NotImplementedError

    def get_all_observations(self):
        """
        Returns a set of all possible observations, if feasible.
        """
        raise NotImplementedError


class RewardModel:
    """
    R(s,a,s') = Pr(r|s,a,s')
    """
    def probability(self, reward, state, action, next_state, **kwargs):
        """
        Args:
            reward(r): float
            state(s) <- basics.State
            action(a) <- basics.Action
            next_state(s') <- basics.State
        Returns:
            float: the probability :math:`\Pr(r|s,a,s')`
        """
        raise NotImplementedError

    def sample(self, state, action, next_state, **kwargs):
        """
        Returns reward randomly sampled according to the distribution of this reward model.
        Args:
            state(s) <- basics.State
            action(a) <- basics.Action
            next_state(s') <- basics.State
        Returns:
            reword(r): float
        """
        raise NotImplementedError

    def argmax(self, state, action, next_state, **kwargs):
        """
        Returns the most likely reward. This is optional.
        """
        raise NotImplementedError

    def get_distribution(self, state, action, next_state, **kwargs):
        """
        Returns the underlying distribution of the model
        """
        raise NotImplementedError


class BlackboxModel:
    """
    A BlackboxModel is the generative distribution G(s,a)
    which can generate samples where each is a tuple (s',o,r).
    Agent has either T,O,R or G. Environment has either T,R or G 
    """
    def sample(self, state, action, **kwargs):
        """
        Sample (s',o,r) ~ G(s',o,r)
        """
        raise NotImplementedError

    def argmax(self, state, action, **kwargs):
        """
        Returns the most likely (s',o,r)
        """
        raise NotImplementedError


class PolicyModel:
    """
    \pi(a|s) or \pi(a|h_t)
    """
    def probability(self, action, state, **kwargs):
        """
        Args:
            action(a) <- basics.Action
            state(s) <- basics.State
        Returns:
            \pi(a|s)
        """
        raise NotImplementedError

    def sample(self, state, **kwargs):
        """
        Returns action randomly sampled according to the distribution of this policy model.
        Args:
            state(s) <- basics.State
        Returns:
            Action
        """
        raise NotImplementedError

    def argmax(self, state, **kwargs):
        """
        Returns the most likely action.
        """
        raise NotImplementedError

    def get_distribution(self, state, **kwargs):
        """
        Returns the underlying distribution of the model.
        """
        raise NotImplementedError

    def get_all_actions(self, *args, **kwargs):
        """
        Returns a set of all possible actions, if feasible.
        """
        raise NotImplementedError

    def update(self, state, next_state, action, **kwargs):
        """
        Policy model may be updated given a (s,a,s') pair.
        """
        pass



class Agent:
    """ An Agent operates in an environment by taking actions, receiving observations, and updating its belief.
    The Agent supplies the class:`TransitionModel`, class:`ObservationModel`, class:`RewardModel`, or class:`BlackboxModel` to the planner or the belief update algorithm.
    Taking actions is the job of a planner(class:`Planner`), and the belief update is the job taken care of by the belief representation or the planner.
    """
    def __init__(self, init_belief,
                 policy_model,
                 transition_model=None,
                 observation_model=None,
                 reward_model=None,
                 blackbox_model=None):

        self._init_belief = init_belief
        self._policy_model = policy_model

        self._transition_model = transition_model
        self._observation_model = observation_model
        self._reward_model = reward_model
        self._blackbox_model = blackbox_model

        # It cannot be the case that both explicit models and blackbox model are None.
        if self._blackbox_model is None:
            assert self._transition_model is not None\
                and self._observation_model is not None\
                and self._reward_model is not None

        # For online planning
        self._cur_belief = init_belief
        self._history = ()

    # getter & setter
    @property
    def history(self):
        """
        Current history
        history are of the form ((a,o),...)
        """
        return self._history

    def update_history(self, real_action, real_observation):
        self._history += ((real_action, real_observation),)

    @property
    def init_belief(self):
        """
        Initial belief distribution
        """
        return self._init_belief

    @property
    def belief(self):
        """
        Current belief distribution
        """
        return self.cur_belief

    @property
    def cur_belief(self):
        return self._cur_belief

    def set_belief(self, belief, prior=False):
        self._cur_belief = belief
        if prior:
            self._init_belief = belief

    def sample_belief(self):
        """
        Returns a state (class:`State`) sampled from the belief.
        """
        return self._cur_belief.random()

    @property
    def observation_model(self):
        return self._observation_model

    @property
    def transition_model(self):
        return self._transition_model

    @property
    def reward_model(self):
        return self._reward_model

    @property
    def policy_model(self):
        return self._policy_model

    @property
    def blackbox_model(self):
        return self._blackbox_model

    @property
    def generative_model(self):
        return self.blackbox_model

    def add_attr(self, attr_name, attr_value):
        """
        add_attr(self, attr_name, attr_value)
        A function that allows adding attributes to the agent.
        Sometimes useful for planners to store agent-specific information.
        """
        if hasattr(self, attr_name):
            raise ValueError("attributes %s already exists for agent." % attr_name)
        else:
            setattr(self, attr_name, attr_value)

    def update(self, real_action, real_observation, **kwargs):
        """
        update(self, real_action, real_observation, **kwargs)
        updates the history and performs belief update"""
        raise NotImplementedError

    @property
    def all_states(self):
        """Only available if the transition model implements
        `get_all_states`."""
        return self.transition_model.get_all_states()

    @property
    def all_actions(self):
        """Only available if the policy model implements
        `get_all_actions`."""
        return self.policy_model.get_all_actions()

    @property
    def all_observations(self):
        """Only available if the observation model implements
        `get_all_observations`."""
        return self.observation_model.get_all_observations()

    def valid_actions(self, *args, **kwargs):
        return self.policy_model.get_all_actions(*args, **kwargs)


class Environment:
    """
    An Environment maintains the true state of the world.
    The Environment is passive. It never observes nor acts.
    """
    def __init__(self, init_state,
                 transition_model,
                 reward_model):
        self._init_state = init_state
        self._transition_model = transition_model
        self._reward_model = reward_model
        self._cur_state = init_state

    @property
    def state(self):
        """Synonym for 'cur_state`"""
        return self.cur_state

    @property
    def cur_state(self):
        """Current state of the environment"""
        return self._cur_state

    @property
    def transition_model(self):
        """The class:`TransitionModel` underlying the environment"""
        return self._transition_model

    @property
    def reward_model(self):
        """The class:`RewardModel` underlying the environment"""
        return self._reward_model

    def state_transition(self, action, execute=True, **kwargs):
        """
        Simulates a state transition given `action`.
        If `execute` is set to True, then the resulting state will be the new current state of the environment.

        Args:
            action (Action): action that triggers the state transition
            execute (bool): If True, the resulting state of the transition will become the current state.

        Returns:
            float or tuple:
                reward as a result of `action` and state transition, if `execute` is True
                (next_state, reward) if `execute` is False.
        """
        next_state, reward, _ = sample_explict_models(self.transition_model, None, self.reward_model,
                                                      self.state, action,
                                                      discount_factor=kwargs.get("discount_factor", 1.0))
        if execute:
            self.apply_transition(next_state)
            return reward
        else:
            return next_state, reward

    def apply_transition(self, next_state):
        """
        Apply the transition, that is, assign current state to be the `next_state`.
        """
        self._cur_state = next_state

    def provide_observation(self, observation_model, action, **kwargs):
        """
        Returns an observation sampled according to 'observation_model(Pr(o|s',a))'
        where s' is current environment state, 'a' is the given `action`.

        Args:
            observation_model (ObservationModel)
            action (Action)

        Returns:
            Observation: an observation sampled from 'Pr(o|s',a)`.
        """
        return observation_model.sample(self.state, action, **kwargs)


class Option(Action):
    """An option is a temporally abstracted action that is defined by (I, pi, B),
    where I is a initiation set, pi is a policy, and B is a termination condition.
    A framework for temporal abstraction in reinforcement learning.
    """
    def initiation(self, state, **kwargs):
        """
        Returns True if the given parameters satisfy the initiation set
        """
        raise NotImplementedError

    def termination(self, state, **kwargs):
        """
        Returns a boolean of whether state satisfies the termination condition.
        Technically returning a float between 0 and 1 is also allowed.
        """
        raise NotImplementedError

    @property
    def policy(self):
        """Returns the policy model (PolicyModel) of this option."""
        raise NotImplementedError

    def sample(self, state, **kwargs):
        """
        Samples an action from this option's policy.
        Convenience function; Can be overriden if don't feel like writing a PolicyModel class.
        """
        return self.policy.sample(state, **kwargs)

    def __eq__(self, other):
        raise NotImplementedError

    def __hash__(self):
        raise NotImplementedError


def sample_explict_models(T, O, R, state, action, discount_factor=1.0):
    """
    T: TransitionModel, O: ObservationModel, R: RewardModel
    """
    nsteps = 0

    if isinstance(action, Option):
        # The action is an option; simulate a rollout of the option
        option = action
        if not option.initiation(state):
            # state is not in the initiation set of the option.
            # This is similar to the case when you are in a particular (e.g. terminal) state and certain action cannot be performed,
            # which will still be added to the PO-MCTS tree because some other state with the same history will allow this action.
            # In this case, that certain action will lead to no state change, no observation, and 0 reward, because nothing happened.
            if O is not None:
                return state, None, 0, 0
            else:
                return state, 0, 0

        reward = 0
        step_discount_factor = 1.0
        while not option.termination(state):
            action = option.sample(state)
            next_state = T.sample(state, action)
            reward += step_discount_factor * R.sample(state, action, next_state)
            step_discount_factor *= discount_factor
            state = next_state
            nsteps += 1
        # sample observation at the end, where action is the last action.
    else:
        next_state = T.sample(state, action)
        reward = R.sample(state, action, next_state)
        nsteps += 1
    if O is not None:
        observation = O.sample(next_state, action)
        return next_state, observation, reward, nsteps
    else:
        return next_state, reward, nsteps

def sample_generative_model(agent, state, action, discount_factor=1.0):
    """
    (s', o, r) ~ G(s, a)
    If the agent has transition/observation models, a black box will be created based on these models
    (i.e. s' and o will be sampled according to these models).

    Args:
        agent (Agent): agent that supplies all the models
        state (State)
        action (Action)
        discount_factor (float): Defaults to 1.0; Only used when action is an class:`Option`.

    Returns:
        tuple: (s', o, r, n_steps)
    """
    if agent.transition_model is None:
        # |TODO: not tested|
        result = agent.generative_model.sample(state, action)
    else:
        result = sample_explict_models(agent.transition_model,
                                       agent.observation_model,
                                       agent.reward_model,
                                       state,
                                       action,
                                       discount_factor)
    return result


class Particles(GenerativeDistribution):
    def __init__(self, particles):
        self._particles = particles
        
    @property
    def particles(self):
        return self._particles

    def __str__(self): # ?
        hist = self.get_histogram()
        hist = [(k,hist[k]) for k in list(reversed(sorted(hist, key=hist.get)))]
        return str(hist)

    def __len__(self):
        return len(self._particles)
    
    def __getitem__(self, value):
        """
        Returns the probability of `value`.
        """        
        belief = 0
        for s in self._particles:
            if s == value:
                belief += 1
        return belief / len(self._particles)
    
    def __setitem__(self, value, prob): # ?
        """
        Sets probability of value to `prob`.
        Assume that value is between 0 and 1
        """        
        particles = [s for s in self._particles if s != value]
        # particles = [s for s in self._particles if s.position.all() != value.position.all()]
        len_not_value = len(particles)
        if len_not_value == 0:
            amount_to_add = 1
        else:
            amount_to_add = int(prob * len_not_value / (1 - prob))
        for i in range(amount_to_add): # Append particles as many as the number of prob
            particles.append(value)
        self._particles = particles
        
    def __hash__(self):
        if len(self._particles) == 0:
            return hash(0)
        else:
            # if the value space is large, a random particle would differentiate enough
            indx = random.randint(0, len(self._particles-1))
            return hash(self._particles[indx])
        
    def __eq__(self, other): # ?
        if not isinstance(other, Particles):
            return False
        else:
            if len(self._particles) != len(other.praticles):
                return False
            value_counts_self = {}
            value_counts_other = {}
            for s in self._particles:
                if s not in value_counts_self:
                    value_counts_self[s] = 0
                value_counts_self[s] += 1
            for s in other.particles:
                if s not in value_counts_self:
                    return False
                if s not in value_counts_other:
                    value_counts_other[s] = 0
                value_counts_other[s] += 1
            return value_counts_self == value_counts_other

    def mpe(self, hist=None):
        """
        Choose a particle that is most likely to be sampled.
        """
        if hist is None:
            hist = self.get_histogram()
        return max(hist, key=hist.get)

    def random(self):
        """
        Randomly choose a particle
        """
        if len(self._particles) > 0:
            return random.choice(self._particles)
        else:
            return None
    

    def add(self, particle):
        """
        Add a particle.
        """
        self._particles.append(particle)

    def get_histogram(self):
        """
        Returns a dictionary from value to probability of the histogram
        """        
        value_counts_self = {}
        for s in self._particles:
            if s not in value_counts_self:
                value_counts_self[s] = 0
            value_counts_self[s] += 1
        for s in value_counts_self:
            value_counts_self[s] = value_counts_self[s] / len(self._particles)
        return value_counts_self

    def get_abstraction(self, state_mapper):
        """
        feeds all particles through a state abstraction function.
        Or generally, it could be any function.
        """
        particles = [state_mapper(s) for s in self._particles]
        return particles

    @classmethod
    def from_histogram(cls, histogram, num_particles=1000):
        """
        Given a Histogram distribution `histogram`, return a particle representation of it, which is an approximation.
        """
        particles = []
        for i in range(num_particles):
            particles.append(histogram.random())
        return Particles(particles)
        

"""
? To update particle belief, there are two possibilities.
Either, an algorithm such as POMCP is used which will update the belief automatically when the planner gets updated;
Or, a standalone particle belief distribution is updated.
In the latter case, the belief update algorithm is as described in :cite:`silver2010monte`
but written explicitly instead of done implicitly when the MCTS tree truncates as in POMCP.
"""
def abstraction_over_particles(particles, state_mapper): # ?
    particles = [state_mapper(s) for s in particles]
    return particles

def particle_reinvigoration(particles, num_particles, state_transform_func=None): # ?
    """
    Note that particles should contain states that have already made the transition as a result of the real action.
    Therefore, they simply form part of the reinvigorated particles.
    At least maintain `num_particles` number of particles. If already have more, then it's ok.
    Args:
        particles <- class:Particle
        num_particles: int
        state_transform_func (State->State) is used to add artificial noise to the reinvigorated particles.
    """
    # If not enough particles, introduce artificial noise to existing particles (reinvigoration)
    new_particles = copy.deepcopy(particles)
    if len(new_particles) == 0:
        raise ValueError("Particle deprivation.")
    if len(new_particles) > num_particles:
        return new_particles
    
    # print("Particle reinvigoration for %d particles" % (num_particles - len(new_particles)))    
    while len(new_particles) < num_particles:
        # need to make a copy otherwise the transform affects states in 'particles'
        next_state = copy.deepcopy(particles.random())
        # Add artificial noise
        if state_transform_func is not None:
            next_state = state_transform_func(next_state)
        new_particles.add(next_state)
    return new_particles


def update_particles_belief(current_particles,
                            real_action, real_observation=None,
                            observation_model=None,
                            transition_model=None,
                            blackbox_model=None,
                            state_transform_func=None):
    """
    This is the second case (update particles belief explicitly);
    Either BlackboxModel is not None, or TransitionModel and ObservationModel are not None.
    Note that you DON'T need to call this function if you are using POMCP.
    |TODO: not tested|

    Args:
        current_particles <- class:Particles
        real_action <- class:Action
        real_Observation <- class:Observation
        observation_model <- classObservationModel
        transition_model <- class:TransitionModel
        blackbox_model <- class:BlackboxModel
        state_transform_func (State->State) is used to add artificial noise to the reinvigorated particles.
    """
    filtered_particles = []
    for particle in current_particles.particles:
        # particle represents a state
        if blackbox_model is not None:
            # We're using a blackbox generator; (s',o,r) ~ G(s,a)
            result = blackbox_model.sample(particle, real_action)
            next_state = result[0]
            observation = result[1]
        else:
            # We're using explicit models
            next_state = transition_model.sample(particle, real_action)
            observation = observation_model.sample(next_state, real_action)
        # If observation matches real, then the next_state is accepted
        if observation == real_observation:
            filtered_particles.append(next_state)
    # Particle reinvigoration
    return particle_reinvigoration(Particles(filtered_particles), len(current_particles.particles),
                                   state_transform_func=state_transform_func)


class Planner:
    """
    A Planner can `plan` the next action to take for a given agent (online planning).
    The planner can be updated (which may also update the agent belief) once an action is executed and observation received.
    """

    def plan(self, agent):    
        """
        The agent carries the information: Bt, ht, O,T,R/G, pi, necessary for planning
        Args:
            agent <- class:Agent
        """
        raise NotImplementedError

    def update(self, agent, real_action, real_observation):
        """
        Updates the planner based on real action and observation.
        Updates the agent accordingly if necessary.
        If the agent's belief is also updated here, the `update_agent_belief` attribute should be set to True. By default, does nothing.
        Args:
            agent <- class:Agent
            real_action <- class:Action
            real_observation <- class:Observation
        """
        pass    

    def updates_agent_belief(self):
        """
        True if planner's update function also updates agent's belief.
        """
        return False


class Histogram(GenerativeDistribution):

    """
    Histogram representation of a probability distribution.
    """

    def __init__(self, histogram):
        """
        Args:
            `histogram(dict)` is a dictionary mapping from variable value to probability
        """
        if not (isinstance(histogram, dict)):
            raise ValueError("Unsupported histogram representation! %s"
                             % str(type(histogram)))
        self._histogram = histogram

    @property
    def histogram(self):
        """histogram(self)"""
        return self._histogram

    def __str__(self):
        if isinstance(self._histogram, dict):
            return str([(k,self._histogram[k])
                        for k in list(reversed(sorted(self._histogram,
                                                      key=self._histogram.get)))[:5]])

    def __len__(self):
        return len(self._histogram)

    def __getitem__(self, value):
        """
        Returns the probability of `value`.
        """
        if value in self._histogram:
            return self._histogram[value]
        else:
            return 0

    def __setitem__(self, value, prob):
        """
        Sets probability of value to `prob`.
        """
        self._histogram[value] = prob

    def __hash__(self):
        if len(self._histogram) == 0:
            return hash(0)
        else:
            # if the domain is large, a random state would differentiate enough
            value = self.random()
            return hash(self._histogram[value])

    def __eq__(self, other):
        if not isinstance(other, Histogram):
            return False
        else:
            return self.histogram == other.histogram

    def __iter__(self):
        return iter(self._histogram)

    def mpe(self):
        """
        Returns the most likely value of the variable.
        """
        return max(self._histogram, key=self._histogram.get)

    def random(self):
        """
        Randomly sample a value based on the probability in the histogram
        """
        candidates = list(self._histogram.keys())
        prob_dist = []
        for value in candidates:
            prob_dist.append(self._histogram[value])
        if sys.version_info[0] >= 3 and sys.version_info[1] >= 6:
            # available in Python 3.6+
            return random.choices(candidates, weights=prob_dist, k=1)[0]
        else:
            return np.random.choice(candidates, 1, p=prob_dist)[0]

    def get_histogram(self):
        """
        Returns a dictionary from value to probability of the histogram
        """
        return self._histogram

    # Deprecated; it's assuming non-log probabilities
    def is_normalized(self, epsilon=1e-9):
        """
        Returns true if this distribution is normalized
        """
        prob_sum = sum(self._histogram[state] for state in self._histogram)
        return abs(1.0-prob_sum) < epsilon
    

def expectation_histogram(hist: Histogram):
    total_weight = 0
    weighted_sum = [0,0]
    for state in hist:
        total_weight += hist[state]
        pos = state.position
        weighted_sum[0] += hist[state]*pos[0]
        weighted_sum[1] += hist[state]*pos[1]
    return weighted_sum/total_weight

def abstraction_over_histogram(current_histogram, state_mapper):
    state_mappings = {s:state_mapper(s) for s in current_histogram}
    hist = {}
    for s in current_histogram:
        a_s = state_mapper(s)
        if a_s not in hist[a_s]:
            hist[a_s] = 0
        hist[a_s] += current_histogram[s]
    return hist

def update_histogram_belief(current_histogram, 
                            real_action, real_observation,
                            observation_model, transition_model, oargs={},
                            targs={}, normalize=True, static_transition=False,
                            next_state_space=None):
    """
    This update is based on the equation:
        :math:`B_{new}(s') = n O(z|s',a) \sum_s T(s'|s,a)B(s)`.
    Args:
        current_histogram: Histogram
            The Histogram that represents current belief.
        real_action: Action
        real_observation: Observation
        observation_model: ObservationModel
        transition_model: TransitionModel
        oargs: dict
            Additional parameters for observation_model (default {})
        targs: dict
            Additional parameters for transition_model (default {})
        normalize: bool
            True if the updated belief should be normalized
        static_transition: bool
            True if the transition_model is treated as static.
            This basically means Pr(s'|s,a) = Indicator(s' == s). This then means that sum_s Pr(s'|s,a)*B(s) = B(s'), since s' and s have the same state space.
            This thus helps reduce the computation cost by avoiding the nested iteration over the state space.
            But still, updating histogram belief requires iteration of the state space, which may already be prohibitive.
        next_state_space: set
            The state space of the updated belief. 
            By default, this parameter is None and the state space given by current_histogram will be directly considered as the state space of the updated belief.
            This is useful for space and time efficiency in problems where the state space contains parts that the agent knows will deterministically update, and thus not keeping track of the belief over these states.
    Returns:
        Histogram: the histogram distribution as a result of the update
    """
    new_histogram = {}  # state space still the same.
    total_prob = 0
    if next_state_space is None:
        next_state_space = current_histogram
    for next_state in next_state_space:
        observation_prob = observation_model.probability(real_observation,
                                                         next_state,
                                                         real_action,
                                                         **oargs)
        if not static_transition:
            transition_prob = 0
            for state in current_histogram:
                transition_prob += transition_model.probability(next_state,
                                                                state,
                                                                real_action,
                                                                **targs) * current_histogram[state]
        else:
            transition_prob = current_histogram[next_state]
            
        new_histogram[next_state] = observation_prob * transition_prob
        total_prob += new_histogram[next_state]

    # Normalize
    if normalize:
        for state in new_histogram:
            if total_prob > 0:
                new_histogram[state] /= total_prob
    return Histogram(new_histogram)


class Gaussian(GenerativeDistribution):

    """Note that Gaussian distribution is defined over a vector of numerical variables,
    but not a State variable.

    __init__(self, mean, cov)"""

    def __init__(self, mean, cov):
        """
        mean (list): mean vector
        cov (list): covariance matrix
        """
        self._mean = mean
        self._cov = cov

    @property
    def mean(self):
        return self._mean

    @property
    def covariance(self):
        return self._cov

    @property
    def cov(self):
        return self.covariance

    @property
    def sigma(self):
        return self.covariance

    def __getitem__(self, value):
        """__getitem__(self, value)
        Evaluate the probability of given value
        
        Args:
            value (list or array like)
        Returns:
            float: The density of multivariate Gaussian."""
        return multivariate_normal.pdf(np.array(value),
                                       np.array(self._mean),
                                       np.array(self._cov))
        
            
    def __setitem__(self, value, prob):
        """__setitem__(self, value, prob)
        Not Implemented;
        It isn't supported to arbitrarily change a value in
        a Gaussian distribution to have a given probability.
        """
        raise NotImplementedError("Gaussian does not support density assignment.")

    def __hash__(self):
        return hash(tuple(self._mean))

    def __eq__(self, other):
        if not isinstance(other, Gaussian):
            return False
        else:
            return self._mean == other.mean\
                and self._cov == other.cov

    def mpe(self):
        """mpe(self)"""
        # The mean is the MPE
        return self._mean

    def random(self, n=1): # ?
        """random(self, n=1)"""
        d = len(self._mean)
        Xstd = np.random.randn(n, d)
        X = np.dot(Xstd, sqrtm(self._cov)) + self._mean
        if n == 1:
            return X.tolist()[0]
        else:
            return X.tolist()
