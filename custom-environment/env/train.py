# """Uses Ray's RLlib to train agents to play Pistonball.

# Author: Rohan (https://github.com/Rohan138)
# """

# import os

# import ray
# import supersuit as ss
# from ray import tune
# from ray.rllib.algorithms.ppo import PPOConfig
# from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
# from ray.rllib.models import ModelCatalog
# from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
# from ray.tune.registry import register_env
# from torch import nn
# from custom_environment import Spec_Ops_Env
# from pettingzoo.butterfly import pistonball_v6

# uri = 'file://' + os.path.abspath('./ray_results/custom_environment_v0')


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


# def env_creator(args):
#     env = Spec_Ops_Env.parallel_env()
#     return env


# if __name__ == "__main__":
#     ray.init()

#     env_name = "custom_environment_v0"

#     register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator(config)))
#     ModelCatalog.register_custom_model("CNNModelV2", CNNModelV2)

#     config = (
#         PPOConfig()
#         .environment(env=env_name, clip_actions=True)
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
#         .debugging(log_level="ERROR")
#         .framework(framework="torch")
#         .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
#     )

#     tune.run(
#         "PPO",
#         name="PPO",
#         stop={"timesteps_total": 5000000},
#         config=config.to_dict(),
#     )

"""Uses Ray's RLlib to train agents to play Pistonball.

Author: Rohan (https://github.com/Rohan138)
"""

import os

import ray
import supersuit as ss
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.tune.registry import register_env
from torch import nn
from custom_environment import Spec_Ops_Env

class CNNModelV2(TorchModelV2, nn.Module):
    def __init__(self, obs_space, act_space, num_outputs, *args, **kwargs):
        TorchModelV2.__init__(self, obs_space, act_space, num_outputs, *args, **kwargs)
        nn.Module.__init__(self)
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, [8, 8], stride=(4, 4)),
            nn.ReLU(),
            nn.Conv2d(32, 64, [4, 4], stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(64, 64, [3, 3], stride=(1, 1)),
            nn.ReLU(),
            nn.Flatten(),
            (nn.Linear(3136, 512)),
            nn.ReLU(),
        )
        self.policy_fn = nn.Linear(512, num_outputs)
        self.value_fn = nn.Linear(512, 1)

    def forward(self, input_dict, state, seq_lens):
        model_out = self.model(input_dict["obs"].permute(0, 3, 1, 2))
        self._value_out = self.value_fn(model_out)
        return self.policy_fn(model_out), state

    def value_function(self):
        return self._value_out.flatten()
    
def env_creator(args):
    env = Spec_Ops_Env()
    
    # env = ss.dtype_v0(env, "int32")
    # env = ss.normalize_obs_v0(env, env_min=0, env_max=1)
    
    return env

if __name__ == "__main__":
    ray.init()

    env_name = "123"

    register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator(config)))
    ModelCatalog.register_custom_model("CNNModelV2", CNNModelV2)

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
            policies=env.get_agent_ids(),
            policy_mapping_fn=(lambda agent_id, *args, **kwargs: agent_id),
        )
        .debugging(log_level="ERROR")
        .framework(framework="torch")
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    )


# Get the user's home directory
user_home = os.path.expanduser("~/RL_Spec_Ops_logs")


# Specify the Downloads directory within the user's home directory
#downloads_dir = os.path.join(user_home, "Downloads")
# Use the downloads directory as the local_dir
# downloads_dir = os.path.join(user_home, "1")
local_dir = user_home

print(local_dir)

tune.run(
        "PPO",
        name="PPO",
        stop={"timesteps_total": 5000000},
        checkpoint_freq=10,
        local_dir=local_dir,
        config=config.to_dict(),
    )