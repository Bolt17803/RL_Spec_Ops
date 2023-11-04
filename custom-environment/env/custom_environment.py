import gymnasium.spaces
import functools
import random
from copy import copy
import math
import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete
from pettingzoo import ParallelEnv
from pettingzoo.test import parallel_api_test
from pettingzoo.test import render_test

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
        self.agents = copy(self.possible_agents)
        self.timestamp=0

        self.terr_x= np.random.randint(0,9)
        self.terr_y= np.random.randint(0,9)
        # self.terr_angle= np.random.randint(0,359)
        self.terr_angle= 0
        self.sol_x=np.random.randint(0,9)
        self.sol_y =np.random.randint(0,9)

        observations = {
            a: (
                self.terr_x + 10 * self.terr_y,
                self.sol_x + 10 * self.sol_y,
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

        if terr_action == 1 and self.terr_x > 0:
            self.terr_x -= 1 # left
        elif terr_action == 2 and self.terr_x < 9:
            self.terr_x += 1 # right
        elif terr_action == 3 and self.terr_y > 0:
            self.terr_y -= 1 # top
        elif terr_action == 4 and self.terr_y < 9:
            self.terr_y += 1 # bottom

        if terr_action == "i" :
            self.terr_angle += 30 # rotate 30 degrees anti clockwise
        elif terr_action == "d" and self.prisoner_x < 9:
            self.prisoner_x -= 30 # rotate 30 degrees clockwise

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
            rewards={"soldier":1, "terrorist":1}
            terminations = {a: True for a in self.agents}

        truncations = {a: False for a in self.agents}
        if self.timestamp > 100:
            rewards = {"prisoner": 0, "guard": 0}
            truncations = {"prisoner": True, "guard": True}
        self.timestamp += 1

        # get observations
        observations = {
            a: (
                self.terr_x + 10 * self.terr_y,
                self.sol_x + 10 * self.sol_y,
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
        grid[self.prisoner_y, self.prisoner_x] = "P"
        grid[self.guard_y, self.guard_x] = "G"
        grid[self.escape_y, self.escape_x] = "E"
        print(f"{grid} \n")

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        return MultiDiscrete([10 * 10] * 3)
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(6)
    
if __name__ == "__main__":
    env = CustomEnvironment()
    parallel_api_test(env, num_cycles=1_000_000)