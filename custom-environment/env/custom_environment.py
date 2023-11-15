import functools
import random
from copy import copy
import math
import numpy as np
import matplotlib.pyplot as plt
import time

import gymnasium
import gymnasium.spaces
from gymnasium.spaces import Discrete, MultiDiscrete

from pettingzoo import AECEnv, ParallelEnv
from pettingzoo.utils import parallel_to_aec
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo.test import parallel_api_test
from pettingzoo.test import render_test

from visualizer import Visualizer

#Default variables
MAP_SIZE = (80, 80)

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

    def __init__(self, render_mode=None, config=None):
        '''
        we initialize the following:
        - the starting location of terrorist
        - the random initialization location of the dummy soldier
        - timestamp
        - possible agents

        these attributes should not be changed after initialization
        '''

        self.config = config or {'empty':None}

        #Initializing
        self.possible_agents=["terrorist_"+str(i) for i in range(self.config.get('num_terr',1))]
        self.possible_agents.extend(["soldier_"+str(i) for i in range(self.config.get('num_sol',1))])
        self.shoot_angle = self.config.get('shoot_angle', 15)   #Common to all agentsagents

        self.terr_x=None    #Change this to support multiple agents
        self.terr_y=None
        self.terr_angle=None
        self.terr_fov = self.config.get('terr_fov', 30) #please keep <=179 so math works correctly!!!

        self.sol_x=None
        self.sol_y=None
        self.sol_angle=None
        self.soldier_fov = self.config.get('sol_fov', 30)  #please keep <=179 so math works correctly!!!

        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        print(self.agent_name_mapping, "::::::")
        self.timestamp=None

        #initializing rendering screen
        self.render_mode = self.config.get('render_mode', 'ansi')    #Check clashing with render_mode variable
        self.map_size = self.config.get('map_size', MAP_SIZE)
        self.viz = Visualizer(grid=self.map_size)

        #Error Checking
        if(self.soldier_fov >= 180 or self.terr_fov >= 180):
            print("invalid fov angle, line 51,52 chusko bey")
            exit()

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        return MultiDiscrete([10 * 10] * 3) #Change this

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(6)

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
                "angle":self.sol_angle,
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

    def close(self):
        """
        CLose releases the pygame graphical display when env is no longer being used.
        """
        self.Viz.quit()
        pass

    def reset(self, seed=None, options=None):
        """
        Reset the environment to the starting point
        it needs to initialize the follownig attributes:
        - agents
        - timestamp
        - terrorist coordinates, angles
        - soldier coordinates, angles
        - observation
        - infos

        and must set up the environment so that render(), step(), and observe() can be called without an issue
        """
        np.random.seed(seed) if seed else print('No Seeding only Determinism!!!!')

        self.agents = self.possible_agents[:]
        self.timestamp=0

        self.terr_x, self.terr_y, self.terr_angle = self.config.get('terr_x',-1), self.config.get('terr_y',-1), self.config.get('terr_angle', -1)  #Error handling for invalid inputs required!
        self.terr_x=np.random.randint(0,9) if self.terr_x<0 else self.terr_x #Randomly place the terrorist on the grid, facing an arbitrary angle
        self.terr_y=np.random.randint(0,9) if self.terr_y<0 else self.terr_y
        self.terr_angle=np.random.randint(0,359) if self.terr_angle<0 else self.terr_angle

        self.sol_x, self.sol_y, self.sol_angle = self.config.get('sol_x',-1), self.config.get('sol_y',-1), self.config.get('sol_angle', -1)
        self.sol_x=np.random.randint(0,9) if self.sol_x<0 else self.terr_x #Randomly place the soldier on the grid, facing an arbitrary angle
        self.sol_y=np.random.randint(0,9) if self.sol_y<0 else self.terr_y
        self.sol_angle=np.random.randint(0,359) if self.sol_angle<0 else self.sol_angle

        infos = {a: {} for a in self.agents}
        observations = {agent: None for agent in self.agents}  #used by step() and observe()
        # self.observations = {
        #     a: (
        #         [self.terr_x, self.terr_y],
        #         [self.sol_x, self.sol_y],
        #         self.terr_angle,
        #         self.sol_angle,
        #     )
        #     for a in self.agents
        # }

        self.state = np.zeros((self.config.get('map_size', MAP_SIZE)))  #{agent: NONE for agent in self.agents} #used by step()

        return observations, infos

    def step(self, actions):
        """
        takes in an action for the current agent (specified by the agent_selection)

        needs to update:
        - coordinates of terrorist
        - coordinates of soldier
        - rotation of the terrorist
        - rotation of the soldier
        - termination condition
        - rewards
        - timestamp
        - infos
        - truncations

        add any internl state  use by observe() or render()
        """

        # If a user passes in actions with no agents, then just return empty observations, etc.
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}


        # execute actions
        self.move(actions)

        # Initialize termination conditions and rewards
        terminations = {a: False for a in self.agents}
        rewards = {a: 0 for a in self.agents}   # rewards for all agents are placed in the rewards dictionary to be returned

        x1, y1=self.terr_x, self.terr_y #terrorist coordinates
        x2, y2=self.sol_x, self.sol_y # soldier coordinates

        #Calculate the rewards and punishments
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
        print("terrorist pov:",tt1,angle_soldier,tt2,self.terr_angle)
        '''the scope of angle is 0<=angle<=359, there is no 360'''
        # if(((angle_soldier >= tt1) and tt2 >= (angle_soldier)) and (tt2>tt1)):
        #     rewards={"soldier":0, "terrorist":1}
        #     terminations = {a: True for a in self.agents}
        # elif((tt2<tt1) and (angle_soldier<=tt2 or angle_soldier>=tt1)):
        #     rewards={"soldier":0, "terrorist":1}
        #     terminations = {a: True for a in self.agents}
        # else:
        #     rewards={"soldier":1, "terrorist":-1}
        reward_t={a: 0 for a in self.agents}
        reward_s={a: 0 for a in self.agents}
        #Terrorist: FOV & Shoot Range Rewards/Punishments
        if tt2>tt1 :
            if ((angle_soldier>=tt1) and (angle_soldier<ss1)): # soldier in between right most shoot and fov line
                reward_t={"soldier":-1, "terrorist":2}
            elif((angle_soldier>=ss1) and (angle_soldier<=ss2)): # soldier in the shooting angle
                reward_t={"soldier":-3, "terrorist":3}
                terminations = {a: True for a in self.agents}
            elif((angle_soldier>ss2) and (angle_soldier<=tt2)):
                reward_t={"soldier":-1, "terrorist":2}
            else:
                reward_t={"soldier":2, "terrorist":-1}
        else:
            if tt1>ss1:
                if (((angle_soldier>=tt1) and (angle_soldier>ss1)) or ((angle_soldier<tt1) and (angle_soldier<ss1))): # soldier in between right most shoot and fov line
                    reward_t={"soldier":-1, "terrorist":2}
                elif((angle_soldier>=ss1) and (angle_soldier<=ss2)): # soldier in the shooting angle
                    reward_t={"soldier":-3, "terrorist":3}
                    terminations = {a: True for a in self.agents}
                elif((angle_soldier>ss2) and (angle_soldier<=tt2)):
                    reward_t={"soldier":-1, "terrorist":2}
                else:
                    reward_t={"soldier":2, "terrorist":-1}
            elif ss1>ss2:
                if ((angle_soldier>=tt1) and (angle_soldier<ss1)): # soldier in between right most shoot and fov line
                    reward_t={"soldier":-1, "terrorist":2}
                elif(((angle_soldier>=ss1) and (angle_soldier>ss2)) or ((angle_soldier<ss1) and (angle_soldier<=ss2))):
                    reward_t={"soldier":-3, "terrorist":3}
                    terminations = {a: True for a in self.agents}
                elif((angle_soldier>ss2) and (angle_soldier<=tt2)):
                    reward_t={"soldier":-1, "terrorist":2}
                else:
                    reward_t={"soldier":2, "terrorist":-1}
            else:
                if ((angle_soldier>=tt1) and (angle_soldier<ss1)): # soldier in between right most shoot and fov line
                    reward_t={"soldier":-1, "terrorist":2}
                elif((angle_soldier>=ss1) and (angle_soldier<=ss2)): # soldier in the shooting angle
                    reward_t={"soldier":-3, "terrorist":3}
                    terminations = {a: True for a in self.agents}
                elif(((angle_soldier>tt2) and (angle_soldier>ss2)) or ((angle_soldier<=tt2) and (angle_soldier<ss2))):
                    reward_t={"soldier":-1, "terrorist":2}
                else:
                    reward_t={"soldier":2, "terrorist":-1}

        #Soldier: FOV & Shoot Range Rewards/Punishments
        tt1_ = tt1  #Saving variables
        tt2_ = tt2
        ss1_ = ss1
        ss2_ = ss2
        angle_soldier_ = angle_soldier

        angle_soldier = angle_from_agent(self.sol_x, self.sol_y, self.terr_x, self.terr_y)
        # right most angles
        ss1 = self.sol_angle-self.shoot_angle/2
        if(ss1<0) : ss1 = 360+ss1
        tt1 = self.sol_angle-self.terr_fov/2
        if(tt1<0): tt1 = 360+tt1
        # left most angles
        ss2 = self.sol_angle+self.shoot_angle/2
        if(ss2>=360): ss2=ss2-360
        tt2 = self.sol_angle+self.terr_fov/2
        if(tt2>=360): tt2=tt2-360
        print("soldier pov:",tt1,angle_soldier,tt2,self.sol_angle)
        if tt2>tt1 :
            if ((angle_soldier>=tt1) and (angle_soldier<ss1)): # terrorist in between right most shoot and fov line
                reward_s={"soldier":2, "terrorist":-1}
            elif((angle_soldier>=ss1) and (angle_soldier<=ss2)): # terrorist in the shooting angle
                reward_s={"soldier":3, "terrorist":-3}
                terminations = {a: True for a in self.agents}
            elif((angle_soldier>ss2) and (angle_soldier<=tt2)):
                reward_s={"soldier":2, "terrorist":-1}
            else:
                reward_s={"soldier":-1, "terrorist":2}
        else:
            if tt1>ss1:
                if (((angle_soldier>=tt1) and (angle_soldier>ss1)) or ((angle_soldier<tt1) and (angle_soldier<ss1))): # soldier in between right most shoot and fov line
                    reward_s={"soldier":2, "terrorist":-1}
                elif((angle_soldier>=ss1) and (angle_soldier<=ss2)): # terrorist in the shooting angle
                    reward_s={"soldier":3, "terrorist":-3}
                    terminations = {a: True for a in self.agents}
                elif((angle_soldier>ss2) and (angle_soldier<=tt2)):
                    reward_s={"soldier":2, "terrorist":-1}
                else:
                    reward_s={"soldier":-1, "terrorist":2}
            elif ss1>ss2:
                if ((angle_soldier>=tt1) and (angle_soldier<ss1)): # terrorist in between right most shoot and fov line
                    reward_s={"soldier":2, "terrorist":-1}
                elif(((angle_soldier>=ss1) and (angle_soldier>ss2)) or ((angle_soldier<ss1) and (angle_soldier<=ss2))):
                    reward_s={"soldier":3, "terrorist":-3}
                    terminations = {a: True for a in self.agents}
                elif((angle_soldier>ss2) and (angle_soldier<=tt2)):
                    reward_s={"soldier":2, "terrorist":-1}
                else:
                    reward_s={"soldier":-1, "terrorist":2}
            else:
                if ((angle_soldier>=tt1) and (angle_soldier<ss1)): 
                    reward_s={"soldier":2, "terrorist":-1}
                elif((angle_soldier>=ss1) and (angle_soldier<=ss2)):
                    reward_s={"soldier":3, "terrorist":-3}
                    terminations = {a: True for a in self.agents}
                elif(((angle_soldier>tt2) and (angle_soldier>ss2)) or ((angle_soldier<=tt2) and (angle_soldier<ss2))):
                    reward_s={"soldier":2, "terrorist":-1}
                else:
                    reward_s={"soldier":-1, "terrorist":2}

        for i in rewards.keys():
            print(i)
            rewards[i]=reward_s[i]+reward_t[i]

        truncations = {a: False for a in self.agents}
        if self.timestamp > 1000:
            rewards = {"soldier": 0, "terrorist": 0}
            truncations = {"soldier": True, "terrorist": True}
        self.timestamp += 1

        # get observations
        observations = self.get_observations()
        # observations = {
        #     a: (
        #         [self.terr_x, self.terr_y],
        #         [self.sol_x, self.sol_y],
        #         self.terr_angle,
        #         self.sol_angle,
        #     )
        #     for a in self.agents
        # }

        # Get dummy infos (not used for now)
        infos = {a: {} for a in self.agents}

        if any(terminations.values()) or all(truncations.values()):
            self.agents = []

        if self.render_mode != None:
            self.render()

        return observations, rewards, terminations, truncations, infos

    def move(self, actions):

        terr_action=actions["terrorist_0"]
        sol_action=actions["soldier_0"]

        self.state[self.terr_x][self.terr_y] = 0
        self.state[self.sol_x][self.sol_y] = 0

        #NOTE: Write logic for walls and other agents in both action failsafe and also action masks
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

        if sol_action == 0 and self.sol_x > 0:
            self.sol_x -= 1 # left
        elif sol_action == 1 and self.sol_x < 9:
            self.sol_x += 1 # right
        elif sol_action == 2 and self.sol_y > 0:
            self.sol_y -= 1 # top
        elif sol_action == 3 and self.sol_y < 9:
            self.sol_y += 1 # bottom
        elif sol_action == 4 :
            self.sol_angle += 30 # rotate 30 degrees anti clockwise
            if self.sol_angle>360:
                self.sol_angle=self.sol_angle-360
        elif sol_action == 5 :
            self.sol_angle -= 30 # rotate 30 degrees clockwise
            if self.sol_angle<0:
                self.sol_angle=360+self.sol_angle

        self.state[self.terr_x][self.terr_y] = 2 # terroristis reprsented with number 2
        self.state[self.sol_x][self.sol_y] = 1 # soldier is represented with number 1

        #Generate Action masks
        terr_action_mask = np.ones(6, dtype=np.int8)
        if self.terr_x == 0:
            terr_action_mask[0] = 0
        if self.terr_x == self.map_size[1]-1:
            terr_action_mask[1] = 0
        if self.terr_y == 0:
            terr_action_mask[2] = 0
        if self.terr_y == self.map_size[0]-1:
            terr_action_mask[3] = 0

        sol_action_mask = np.ones(6, dtype=np.int8)
        if self.sol_x == 0:
            sol_action_mask[0] = 0
        if self.sol_x == self.map_size[1]-1:
            sol_action_mask[1] = 0
        if self.sol_y == 0:
            sol_action_mask[2] = 0
        if self.sol_y == self.map_size[0]-1:
            sol_action_mask[3] = 0


    
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

if __name__ == '__main__':
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
