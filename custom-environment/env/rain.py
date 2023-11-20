from custom_environment import Spec_Ops_Env
env = Spec_Ops_Env()

import gym

# Assuming 'env' is your multi-agent environment instance
env = gym.make('env')

# Reset the environment to start a new episode
observations, infos = env.reset()

# Loop through each agent in the environment
for agent in env.agents:
    # Access action space for the current agent
    action_space = env.action_space(agent)

    # Take a random action from the action space for the current agent
    action = action_space.sample()

    # Perform the action in the environment
    new_observations, rewards, terminations, truncations, infos = env.step({agent: action})

