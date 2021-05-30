from dataclasses import dataclass

import matplotlib.pyplot as plt

from POMDP_framework import *
from POMCP import *
from POMCPOW import *
from light_dark_problem import *


@dataclass
class Setting():
    num_simulation: int = 1000
    num_particles: int = 1000
    std_init_belief: float = 0.1
    random_range: float = 1.0
    planning_horizon: int = 30
    discont_factor: float = 1.0
    light: float = 5.0
    const: float = 0.001
    planning_time: float = -1.0
    exploration_const: float = math.sqrt(2)
    num_visits_init: int = 0
    value_init: int = 0


def main():
    opt = Setting()

    init_state = State((2.5, 2.5))
    goal_state = State(tuple(opt.random_range * (np.random.rand(2) - 0.5)))

    init_belief = []
    while len(init_belief) < opt.num_particles:
        sample = State(tuple(np.asarray(init_state.position) + opt.std_init_belief * (np.random.randn(2))))
        init_belief.append(sample)
    init_belief = Particles(init_belief)

    # creates POMDP model
    light_dark_problem = LightDarkProblem(init_state, init_belief, goal_state, opt.light, opt.const)
    light_dark_problem.agent.set_belief(init_belief)

    # set planner
    planner = POMCPOW(pomdp=light_dark_problem, max_depth=opt.planning_horizon, planning_time=opt.planning_time, num_sims=opt.num_simulation,
                    discount_factor=opt.discont_factor, exploration_const=opt.exploration_const,
                    num_visits_init=opt.num_visits_init, value_init=opt.value_init)

    # Visualization setting
    x_range = (-2, 6)
    y_range = (-2, 4)
    viz = LightDarkViz(light_dark_problem, x_range, y_range, 0.1)

    viz.log_state(light_dark_problem.env.state)
    viz.log_belief_expectation(expectation_belief(light_dark_problem.agent.cur_belief))

    test_action = (2.501, 0)

    next_state = light_dark_problem.agent.transition_model.sample(light_dark_problem.env.state, test_action)
    real_observation = light_dark_problem.agent.observation_model.sample(next_state, test_action)
    
    new_belief, prediction = bootstrap_filter(light_dark_problem.agent.belief, next_state, test_action, real_observation, light_dark_problem.agent.observation_model, light_dark_problem.agent.transition_model, len(light_dark_problem.agent.init_belief))
    light_dark_problem.agent.set_belief(new_belief)
    
    viz.log_state(light_dark_problem.env.state)
    viz.log_belief(light_dark_problem.agent.cur_belief)
    viz.log_belief_expectation(expectation_belief(light_dark_problem.agent.cur_belief))

    viz.plot(path_colors={0: [(0,0,0), (0,255,0)],
                            1: [(0,0,0), (255,0,0)]},
                path_styles={0: "--",
                            1: "-"},
                path_widths={0: 4,
                            1: 1})

    plt.show()


if __name__ == '__main__':
    main()