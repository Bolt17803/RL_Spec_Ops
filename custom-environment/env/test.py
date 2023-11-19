# # """Uses Ray's RLlib to train agents to play Pistonball.

# # Author: Rohan (https://github.com/Rohan138)
# # """

# # import os

# # import ray
# # import supersuit as ss
# # from ray import tune
# # from ray.rllib.algorithms.ppo import PPOConfig
# # from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
# # from ray.rllib.models import ModelCatalog
# # from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
# # from ray.tune.registry import register_env
# # from torch import nn
# # from custom_environment import Spec_Ops_Env
# # from pettingzoo.butterfly import pistonball_v6

# # uri = 'file://' + os.path.abspath('./ray_results/custom_environment_v0')


# # class CNNModelV2(TorchModelV2, nn.Module):
# #     def __init__(self, obs_space, act_space, num_outputs, *args, **kwargs):
# #         TorchModelV2.__init__(self, obs_space, act_space, num_outputs, *args, **kwargs)
# #         nn.Module.__init__(self)
# #         self.model = nn.Sequential(
# #             nn.Conv2d(3, 32, [8, 8], stride=(4, 4)),
# #             nn.ReLU(),
# #             nn.Conv2d(32, 64, [4, 4], stride=(2, 2)),
# #             nn.ReLU(),
# #             nn.Conv2d(64, 64, [3, 3], stride=(1, 1)),
# #             nn.ReLU(),
# #             nn.Flatten(),
# #             (nn.Linear(3136, 512)),
# #             nn.ReLU(),
# #         )
# #         self.policy_fn = nn.Linear(512, num_outputs)
# #         self.value_fn = nn.Linear(512, 1)

# #     def forward(self, input_dict, state, seq_lens):
# #         model_out = self.model(input_dict["obs"].permute(0, 3, 1, 2))
# #         self._value_out = self.value_fn(model_out)
# #         return self.policy_fn(model_out), state

# #     def value_function(self):
# #         return self._value_out.flatten()


# # def env_creator(args):
# #     env = Spec_Ops_Env.parallel_env()
# #     return env


# # if __name__ == "__main__":
# #     ray.init()

# #     env_name = "custom_environment_v0"

# #     register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator(config)))
# #     ModelCatalog.register_custom_model("CNNModelV2", CNNModelV2)

# #     config = (
# #         PPOConfig()
# #         .environment(env=env_name, clip_actions=True)
# #         .rollouts(num_rollout_workers=4, rollout_fragment_length=128)
# #         .training(
# #             train_batch_size=512,
# #             lr=2e-5,
# #             gamma=0.99,
# #             lambda_=0.9,
# #             use_gae=True,
# #             clip_param=0.4,
# #             grad_clip=None,
# #             entropy_coeff=0.1,
# #             vf_loss_coeff=0.25,
# #             sgd_minibatch_size=64,
# #             num_sgd_iter=10,
# #         )
# #         .debugging(log_level="ERROR")
# #         .framework(framework="torch")
# #         .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
# #     )

# #     tune.run(
# #         "PPO",
# #         name="PPO",
# #         stop={"timesteps_total": 5000000},
# #         config=config.to_dict(),
# #     )

# """Uses Ray's RLlib to train agents to play Pistonball.

# Author: Rohan (https://github.com/Rohan138)
# """

# import os

# import torch
# import ray
# import supersuit as ss
# from ray import tune
# from ray.rllib.algorithms.ppo import PPOConfig
# from ray.rllib.algorithms.ppo import PPO
# from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
# from ray.rllib.models import ModelCatalog
# from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
# from ray.tune.registry import register_env
# from torch import nn
# from custom_environment import Spec_Ops_Env

# class CNNModelV2(TorchModelV2, nn.Module):
#     def __init__(self, obs_space, act_space, num_outputs, *args, **kwargs):
#         TorchModelV2.__init__(self, obs_space, act_space, num_outputs, *args, **kwargs)
#         nn.Module.__init__(self)
#         self.model = nn.Sequential(
#             nn.Conv2d(3, 32, [8, 8], stride=(4, 4)),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, [4, 4], stride=(2, 2)),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, [3, 3], stride=(1, 1)),
#             nn.ReLU(),
#             nn.Flatten(),
#             (nn.Linear(3136, 512)),
#             nn.ReLU(),
#         )
#         self.policy_fn = nn.Linear(512, num_outputs)
#         self.value_fn = nn.Linear(512, 1)

#     def forward(self, input_dict, state, seq_lens):
#         model_out = self.model(input_dict["obs"].permute(0, 3, 1, 2))
#         self._value_out = self.value_fn(model_out)
#         return self.policy_fn(model_out), state

#     def value_function(self):
#         return self._value_out.flatten()
    
# def env_creator():
#     env = Spec_Ops_Env()
    
#     # env = ss.dtype_v0(env, "int32")
#     # env = ss.normalize_obs_v0(env, env_min=0, env_max=1)
    
#     return env

# def policy_map_fn(agent_id: str, _episode=None, _worker=None, **_kwargs) -> str:
#     """
#     Maps agent_id to policy_id
#     """
#     return agent_id
#     #raise RuntimeError(f'Nee Yabba!!! Invalid agent_id: {agent_id}')
    
# if __name__ == "__main__":
#     ray.init(ignore_reinit_error=True, num_gpus=1)

#     env_name = "123"

#     print("\n\n\nNEE YABBA TORCH CUDA UNDA?:", torch.cuda.is_available(),'\n\n\n')
#     register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator()))
#     ModelCatalog.register_custom_model("CNNModelV2", CNNModelV2)


# env=env_creator()

# config = (
#         PPOConfig()
#         .environment(env="123", clip_actions=True)
#         .rollouts(num_rollout_workers=4, rollout_fragment_length=128)
#         .training(
#             train_batch_size=512,
#             lr=2e-5,
#             gamma=0.99,
#             lambda_=0.9,
#             use_gae=True,
#             clip_param=0.4,
#             grad_clip=None,
#             entropy_coeff=0.1,
#             vf_loss_coeff=0.25,
#             sgd_minibatch_size=64,
#             num_sgd_iter=10,
#         )
#         .multi_agent(
#             policies=env.possible_agents,
#             policy_mapping_fn=(lambda agent_id, *args, **kwargs: agent_id),
#         )
#         .debugging(log_level="ERROR")
#         .framework(framework="torch")
#         .resources(num_gpus=1)#int(os.environ.get("RLLIB_NUM_GPUS", "0"))
#     )


# # Get the user's home directory
# user_home = os.path.expanduser("~/RL_Spec_Ops_logs")


# # Specify the Downloads directory within the user's home directory
# #downloads_dir = os.path.join(user_home, "Downloads")
# # Use the downloads directory as the local_dir
# # downloads_dir = os.path.join(user_home, "1")
# local_dir = user_home

# print(local_dir)

# tune.run(
#         "PPO",
#         name="PPO",
#         stop={"timesteps_total": 5000000},
#         checkpoint_freq=10,
#         local_dir=local_dir,
#         config=config.to_dict(),
#     )

# # from ray.rllib.algorithms.ppo import PPO
# # algo = PPO(config=config, env=env)
# # algo.restore('/home/hemanthgaddey/RL_Spec_Ops_logs/PPO/PPO_123_a44ad_00000_0_2023-11-19_04-54-38/checkpoint_000006/')

# # from ray.rllib.algorithms.algorithm import Algorithm
# # algo = Algorithm.from_checkpoint('/home/hemanthgaddey/RL_Spec_Ops_logs/PPO/PPO_123_a44ad_00000_0_2023-11-19_04-54-38/checkpoint_000006/')

# # obs = env.reset()

# # terminated = truncated = False

# # while not terminated and not truncated:
# #     action = algo.compute_single_action(obs)
# #     obs, reward, terminated, info = env.step(action)
# #     episode_reward += reward
# #     print(episode_reward)

# '''
# # Load the latest saved model
# trained_config = config.copy()
# #trained_config["model"]["custom_model"] = "CNNModelV2"
# trained_trainer = PPOConfig().environment(env="123", clip_actions=True)
# trained_trainer.restore('/home/hemanthgaddey/RL_Spec_Ops_logs/PPO/PPO_123_a44ad_00000_0_2023-11-19_04-54-38/checkpoint_000006/')
# '''
# from ray.rllib.algorithms.algorithm import Algorithm

# # Use the Algorithm's `from_checkpoint` utility to get a new algo instance
# # that has the exact same state as the old one, from which the checkpoint was
# # created in the first place:
# my_new_ppo = Algorithm.from_checkpoint('/home/hemanthgaddey/RL_Spec_Ops_logs/PPO/PPO_123_a44ad_00000_0_2023-11-19_04-54-38/checkpoint_000006/')

# # Function to test the trained model with multiple policies
# my_new_ppo.sample()

# def test_model(my_new_ppo):
#     env = env_creator()
#     obs = env.reset()
#     done = False
#     total_rewards = {policy_id: 0 for policy_id in env.possible_agents}

#     while not done:
#         # Compute actions for each policy
#         actions = {}
#         for policy_id in env.possible_agents:
#             actions[policy_id] = my_new_ppo.compute_actions(obs, policy_id=policy_id)
        
#         obs, rewards, done, _ = env.step(actions)
        
#         # Update total rewards for each policy
#         for policy_id, reward in rewards.items():
#             total_rewards[policy_id] += reward

#     print("Total rewards:")
#     for policy_id, reward in total_rewards.items():
#         print(f"{policy_id}: {reward}")

# # Test the trained model with multiple policies
# test_model(my_new_ppo)

import os

import torch
import ray
import supersuit as ss
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.tune.registry import register_env
from torch import nn
from custom_environment import Spec_Ops_Env
    
def env_creator():
    env = Spec_Ops_Env()    
    return env

def policy_map_fn(agent_id: str, _episode=None, _worker=None, **_kwargs) -> str:
    return agent_id

ray.init(ignore_reinit_error=True, num_gpus=1)

env_name = "123"

print("\n\n\nNEE YABBA TORCH CUDA UNDA?:", torch.cuda.is_available(),'\n\n\n')
register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator()))

env=env_creator()

config = (
        PPOConfig()
        .environment(env="123", clip_actions=True)
        .rollouts(num_rollout_workers=4, rollout_fragment_length=128)
        .training(
            train_batch_size=512,
            lr=2e-5,
            gamma=0.99,
            lambda_=0.9,
            use_gae=True,
            clip_param=0.4,
            grad_clip=None,
            entropy_coeff=0.1,
            vf_loss_coeff=0.25,
            sgd_minibatch_size=64,
            num_sgd_iter=10,
        )
        .multi_agent(
            policies=env.possible_agents,
            policy_mapping_fn=(lambda agent_id, *args, **kwargs: agent_id),
        )
        .debugging(log_level="ERROR")
        .framework(framework="torch")
        .resources(num_gpus=1)#int(os.environ.get("RLLIB_NUM_GPUS", "0"))
    )

from ray.rllib.algorithms.algorithm import Algorithm

def new_policy_mapping_fn(agent_id, episode, worker, **kwargs):
    print(agent_id)
    return  agent_id


algo_w_2_policies = Algorithm.from_checkpoint(
    checkpoint='/home/hemanthgaddey/RL_Spec_Ops_logs/PPO/PPO_123_a44ad_00000_0_2023-11-19_04-54-38/checkpoint_000006/',
    policy_ids=["terrorist_0", "soldier_0"],  # <- restore only those policy IDs here.
    policy_mapping_fn=policy_map_fn,  # <- use this new mapping fn.
)

obs=env.reset()

import time
env.reset()
while True:
    print(obs)
    if(type(obs) == type(())):
        terr_a = algo_w_2_policies.compute_single_action(obs[0]['terrorist_0'], policy_id="terrorist_0")
        sol_a = algo_w_2_policies.compute_single_action(obs[0]['soldier_0'], policy_id="soldier_0")
    else:
        terr_a = algo_w_2_policies.compute_single_action(obs['terrorist_0'], policy_id="terrorist_0")
        sol_a = algo_w_2_policies.compute_single_action(obs['soldier_0'], policy_id="soldier_0")
    obs, rewards, terminations, truncations, infos = env.step({"terrorist_0": terr_a, "soldier_0": sol_a})
    #out.clear_output(wait=True)
    time.sleep(0.1)
    env.render()

    if any(terminations.values()) or all(truncations.values()):
        break
ray.shutdown()
exit()