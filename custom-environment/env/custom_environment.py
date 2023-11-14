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

def angle_from_agent(px, py, sx, sy): # (px,py) are the coordinates from which (sx,sy) angle is measured
    x = sx - px
    y = sy - py
    angle = math.atan2(y, x)
    if angle < 0:
        angle += 2 * math.pi
    return 360-180*(1/math.pi)*angle

class Spec_Ops_Env(ParallelEnv):
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
        self.ya1 = 0
        self.ya2 = 0
        #initializing rendering screen
        self.viz = Visualizer()
        self.terr_fov = 30 #please keep <=179 so math works correctly!!! (we suck at coding)
        self.soldier_fov = 30  #please keep <=179 so math works correctly!!! (we suck at coding)
        self.shoot_angle = 15
        if(self.soldier_fov > 180 or self.terr_fov > 180):
            print("invalid fov angle, line 51 chusko bey")
            exit()

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

        self.terr_x=np.random.randint(0,9)
        self.terr_y=np.random.randint(0,9)
        # self.terr_angle= np.random.randint(0,359)
        self.terr_angle= 315
        self.sol_x=np.random.randint(0,9)
        self.sol_y=np.random.randint(0,9)

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

        elif terr_action == 4 :
            self.terr_angle += 30 # rotate 30 degrees anti clockwise
            if self.terr_angle>360:
                self.terr_angle=self.terr_angle-360
        elif terr_action == 5 :
            self.terr_angle -= 30 # rotate 30 degrees clockwise
            if self.terr_angle<0:
                self.terr_angle=360+self.terr_angle

        # check termination conditions
        terminations = {a: False for a in self.agents}
        rewards = {a: 0 for a in self.agents}

        x1, y1=self.terr_x, self.terr_y #terrorist coordinates
        x2, y2=self.sol_x, self.sol_y # soldier coordinates

        angle_soldier = angle_from_agent(self.terr_x, self.terr_y, self.sol_x, self.sol_y)
        # right most angles
        ss1 = self.terr_angle-self.shoot_angle/2
        if(ss1<0) : ss1 = 360+ss1
        tt1 = self.terr_angle-self.terr_fov/2
        if(tt1<0): tt1 = 360+tt1
        # left most angles
        ss2 = self.terr_angle+self.shoot_angle/2
        if(ss2>=360): ss2=ss2-360
        tt2 = self.terr_angle+self.terr_fov/2
        if(tt2>=360): tt2=tt2-360
        print(tt1,angle_soldier,tt2,self.terr_angle)
        '''the scope of angle is 0<=angle<=359, there is no 360'''
        # if(((angle_soldier >= tt1) and tt2 >= (angle_soldier)) and (tt2>tt1)):
        #     rewards={"soldier":0, "terrorist":1}
        #     terminations = {a: True for a in self.agents}
        # elif((tt2<tt1) and (angle_soldier<=tt2 or angle_soldier>=tt1)):
        #     rewards={"soldier":0, "terrorist":1}
        #     terminations = {a: True for a in self.agents}
        # else:
        #     rewards={"soldier":1, "terrorist":-1}
        if tt2>tt1 :
            if ((angle_soldier>=tt1) and (angle_soldier<ss1)): # soldier in between right most shoot and fov line
                rewards={"soldier":-1, "terrorist":1}
            elif((angle_soldier>=ss1) and (angle_soldier<=ss2)): # soldier in the shooting angle
                rewards={"soldier":-2, "terrorist":2}
                terminations = {a: True for a in self.agents}
            elif((angle_soldier>ss2) and (angle_soldier<=tt2)):
                rewards={"soldier":-1, "terrorist":1}
            else:
                rewards={"soldier":2, "terrorist":-1}
        else:
            if tt1>ss1:
                if (((angle_soldier>=tt1) and (angle_soldier>ss1)) or ((angle_soldier<tt1) and (angle_soldier<ss1))): # soldier in between right most shoot and fov line
                    rewards={"soldier":-1, "terrorist":1}
                elif((angle_soldier>=ss1) and (angle_soldier<=ss2)): # soldier in the shooting angle
                    rewards={"soldier":-2, "terrorist":2}
                    terminations = {a: True for a in self.agents}
                elif((angle_soldier>ss2) and (angle_soldier<=tt2)):
                    rewards={"soldier":-1, "terrorist":1}
                else:
                    rewards={"soldier":2, "terrorist":-1}
            elif ss1>ss2:
                if ((angle_soldier>=tt1) and (angle_soldier<ss1)): # soldier in between right most shoot and fov line
                    rewards={"soldier":-1, "terrorist":1}
                elif(((angle_soldier>=ss1) and (angle_soldier>ss2)) or ((angle_soldier<ss1) and (angle_soldier<=ss2))):
                    rewards={"soldier":-2, "terrorist":2}
                    terminations = {a: True for a in self.agents}
                elif((angle_soldier>ss2) and (angle_soldier<=tt2)):
                    rewards={"soldier":-1, "terrorist":1}
                else:
                    rewards={"soldier":2, "terrorist":-1}
            else:
                if ((angle_soldier>=tt1) and (angle_soldier<ss1)): # soldier in between right most shoot and fov line
                    rewards={"soldier":-1, "terrorist":1}
                elif((angle_soldier>=ss1) and (angle_soldier<=ss2)): # soldier in the shooting angle
                    rewards={"soldier":-2, "terrorist":2}
                    terminations = {a: True for a in self.agents}
                elif(((angle_soldier>tt2) and (angle_soldier>ss2)) or ((angle_soldier<=tt2) and (angle_soldier<ss2))):
                    rewards={"soldier":-1, "terrorist":1}
                else:
                    rewards={"soldier":2, "terrorist":-1}

        # elif(slope1<270 or slope2>90):
        #     if ya1<y2 and y2<ya2:
        #         rewards={"soldier":0, "terrorist":1}
        #         terminations = {a: True for a in self.agents}
        #     else:
        #         rewards={"soldier":1, "terrorist":-1}
        # else:
        #     rewards={"soldier":1, "terrorist":-1}
        # self.rewards = rewards

        truncations = {a: False for a in self.agents}
        if self.timestamp > 1000:
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
        grid[self.terr_y,self.terr_x] = "T"
        grid[self.sol_y,self.sol_x] = "S"

        # grid[self.escape_y, self.escape_x] = "E"

        print(grid)

        state = {
            "m1":{
                "species": "seal",
                "pos":{"x":self.sol_x, "y":self.sol_y},
                "angle":0,
                "fov":self.soldier_fov,
                "shoot_angle":self.shoot_angle,
                "status": "alive"
            },
            "t1":{
                "species": "terrorist",
                "pos":{"x":self.terr_x,"y":self.terr_y},
                "angle":self.terr_angle,
                "fov":self.terr_fov,
                "shoot_angle":self.shoot_angle,
                "status": "alive"
            },
        }
        self.viz.update(state, rewards)
        time.sleep(0.3)


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

env=Spec_Ops_Env()

observations, infos = env.reset()

while env.agents:
    print(env.agents)
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