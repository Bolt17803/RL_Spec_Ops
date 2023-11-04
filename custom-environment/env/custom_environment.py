import gymnasium.spaces
import functools
import random
from copy import copy
import math
import numpy as np
import matplotlib.pyplot as plt
from gymnasium.spaces import Discrete, MultiDiscrete
from pettingzoo import ParallelEnv
from pettingzoo.test import parallel_api_test
from pettingzoo.test import render_test
from visualizer import Visualizer
import time

class CustomEnvironment():
    metadata={
        "name":"custom_environment_v0",
    }

    def __init__(self):
        '''
        we initialize the following:
        - the starting location of terrorist
        - the random initialization location of the dummy soldier
        - timestamp
        - possible agents

        these attributes should not be changed after initialization
        '''
        self.terr_x=None
        self.terr_y=None
        self.terr_angle=None
        self.sol_x=None
        self.sol_y=None
        self.timestamp=None
        self.possible_agents=["terrorist", "soldier"]

        #initializing rendering screen
        self.viz = Visualizer()

    def reset(self, seed=None, options=None):
        """
        Reset the environment to the starting point
        it needs to initialize the follownig attributes:
        - agents
        - timestamp
        - terrorist coordinates
        - soldier coordinates
        - observation
        - infos

        and must set up the environment so that render(), step(), and observe() can be called without an issue
        """
        self.agents = self.possible_agents[:]
        self.timestamp=0

        self.terr_x= np.random.randint(0,9)
        self.terr_y= np.random.randint(0,9)
        # self.terr_angle= np.random.randint(0,359)
        self.terr_angle= 0
        self.sol_x=np.random.randint(0,9)
        self.sol_y =np.random.randint(0,9)

        observations = {
            a: (
                [self.terr_x, self.terr_y],
                [self.sol_x, self.sol_y],
                self.terr_angle,
            )
            for a in self.agents
        }

        infos = {a: {} for a in self.agents}

        return observations, infos

    def step(self, actions):
        """
        takes in an action for the current agent (specified by the agent_selection)

        needs to update:
        - coordinates of terrorist
        - coordinates of soldier
        - rotation of the terrorist
        - termination condition
        - rewards
        - timestamp
        - infos
        - truncations

        add any internl state  use by observe() or render()
        """
        # execute actions
        terr_action=actions["terrorist"]
        sol_action=actions["soldier"]

        if terr_action == 0 and self.terr_x > 0:
            self.terr_x -= 1 # left
        elif terr_action == 1 and self.terr_x < 9:
            self.terr_x += 1 # right
        elif terr_action == 2 and self.terr_y > 0:
            self.terr_y -= 1 # top
        elif terr_action == 3 and self.terr_y < 9:
            self.terr_y += 1 # bottom

        if terr_action == 4 :
            self.terr_angle += 30 # rotate 30 degrees anti clockwise
        elif terr_action == 5 :
            self.terr_angle -= 30 # rotate 30 degrees clockwise

        # check termination conditions
        terminations = {a: False for a in self.agents}
        rewards = {a: 0 for a in self.agents}

        x1, y1=self.terr_x, self.terr_y #terrorist coordinates
        x2, y2=self.sol_x, self.sol_y # soldier coordinates

        # the field of view for the terrorist will be +-30 degrees
        slope2 = self.terr_angle-30
        slope1 = self.terr_angle+30
        slope1=math.tan(math.radians(slope1))
        slope2=math.tan(math.radians(slope2))

        c1 = y1-slope1*x1
        c2 = y2-slope2*x2

        ya1=slope1*x2+c1 # soldier with respect to line one +30 degrees
        ya2=slope2*x2+c2 # soldier with respect to line two -30 degrees

        if y2>ya2 and y2<ya1:
            rewards={"soldier":0, "terrorist":1}
            terminations = {a: True for a in self.agents}
        else:
            rewards={"soldier":1, "terrorist":-1}

        truncations = {a: False for a in self.agents}
        if self.timestamp > 100:
            rewards = {"soldier": 0, "terrorist": 0}
            truncations = {"soldier": True, "terrorist": True}
        self.timestamp += 1

        # get observations
        observations = {
            a: (
                [self.terr_x, self.terr_y],
                [self.sol_x, self.sol_y],
                self.terr_angle,
            )
            for a in self.agents
        }

        # Get dummy infos (not used in this example)
        infos = {a: {} for a in self.agents}

        if any(terminations.values()) or all(truncations.values()):
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def render(self):
        """Renders the environment."""
        grid = np.full((10, 10), " ")
        # print("soldier coordinates:", [self.sol_x, self.sol_y])
        # print("terrorist coordinates:", [self.terr_x, self.terr_y])
        # print("terrorost angle:", self.terr_angle)
        grid[self.terr_x, self.terr_y] = "T"
        grid[self.sol_x, self.sol_y] = "S"

        # grid[self.escape_y, self.escape_x] = "E"

        print(grid)

        state = {
            "m1":{
                "species": "seal",
                "pos":{"x":self.sol_x, "y":self.sol_y},
                "angle":0,
                "status": "alive"
            },
            "t1":{
                "species": "terrorist",
                "pos":{"x":self.terr_x,"y":self.terr_y},
                "angle":self.terr_angle,
                "status": "alive"
            },
        }
        self.viz.update(state)
        time.sleep(0.1)


    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        return MultiDiscrete([10 * 10] * 3)
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(6)
    
    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pass
    
def visualize_environment(env, observations):
    """Visualizes the environment.

    Args:
        env: The environment to visualize.
        observations: The observations from the environment.
    """

    # Create a grid to represent the environment.
    grid = np.zeros((10, 10))

    # Place the agents on the grid.
    for agent, observation in observations.items():
        terr_corr, sol_corr, angle = observation
        if agent=="terrorist":
            grid[terr_corr[1], 9-terr_corr[0]] = 1
        else:
            grid[sol_corr[1], 9-sol_corr[0]] = 2

        # Plot the agent's field of view.
        field_of_view = np.linspace(angle - 30, angle + 30, 100)
        x_points = (9-terr_corr[0]) + np.cos(field_of_view)
        y_points = terr_corr[0] + np.sin(field_of_view)

        plt.plot(x_points, y_points, color="gray")

    # Plot the grid.
    plt.imshow(grid)
    plt.show()

env=CustomEnvironment()

observations, infos = env.reset()

while env.agents:
    # this is where you would insert your policy
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    print(actions)
    observations, rewards, terminations, truncations, infos = env.step(actions)
    env.render()
    print("observations:", observations)
    print("rewards:", rewards)
    print("----------------------------------------")
    # visualize_environment(env, observations)
env.close()
# while env.agents:
#   # Get the observations from the environment.
#   observations = env.step(actions)

#   # Visualize the environment.
#   visualize_environment(env, observations)


# import parallel_rps

# env = parallel_rps.CustomEnvironment(render_mode="human")
# observations, infos = env.reset()

# while env.agents:
#     # this is where you would insert your policy
#     actions = {agent: env.action_space(agent).sample() for agent in env.agents}

#     observations, rewards, terminations, truncations, infos = env.step(actions)
# env.close()
