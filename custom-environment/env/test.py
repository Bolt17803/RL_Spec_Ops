
from ray.rllib.policy.policy import PolicySpec
from custom_environment import Spec_Ops_Env
from gym import spaces
from typing import Dict
import ray
from ray import tune
# from trainer import Trainer
from ray.rllib.algorithms.ppo import PPO

def policy_map_fn(agent_id: str, _episode=None, _worker=None, **_kwargs) -> str:
    """
    Maps agent_id to policy_id
    """
    if 'soldier' in agent_id:
        return 'soldier_policy'
    elif 'terrorist' in agent_id:
        return 'terrorist_policy'
    else:
        raise RuntimeError(f'Invalid agent_id: {agent_id}')
    
config = PPO.get_default_config()
# config.update(
#     {
#     # The batch size collected for each worker
#     "rollout_fragment_length": 1000,
#     # Can be "complete_episodes" or "truncate_episodes"
#     "batch_mode": "complete_episodes",
#     "simple_optimizer": True,
#     "framework": "torch",
#     })
def get_multiagent_policies() -> Dict[str,PolicySpec]:
    policies: Dict[str,PolicySpec] = {}  # policy_id to policy_spec

    policies['soldier_policy'] = PolicySpec(
                policy_class=None, # use default in trainer, or could be YourHighLevelPolicy
                observation_space=Spec_Ops_Env.observation_space['soldier_0'],
                action_space=spaces.Discrete(6),
                config={}
    )

    policies['terrorist_policy'] = PolicySpec(
        policy_class=None,  # use default in trainer, or could be YourLowLevelPolicy
        observation_space=Spec_Ops_Env.observation_space['terrorist_0'],
        action_space=spaces.Discrete(6),
        config={}
    )

    return policies
if __name__=="__main__":
    ray.init(local_mode=True)  # in local mode you can debug it

    RUN_WITH_TUNE = True
    NUM_ITERATIONS = 500  # 500 results in Tensorboard shown with 500 iterations (about an hour)

    # Tune is the system for keeping track of all of the running jobs, originally for
    # hyperparameter tuning
    if RUN_WITH_TUNE:

        tune.registry.register_trainable("YourTrainer", PPO)
        stop = {
                "training_iteration": NUM_ITERATIONS  # Each iteration is some number of episodes
            }
        results = tune.run("YourTrainer", stop=stop, config=config, verbose=1, checkpoint_freq=10)

        # You can just do PPO or DQN but we wanted to show how to customize
        #results = tune.run("PPO", stop=stop, config=config, verbose=1, checkpoint_freq=10)

    else:
        from custom_environment import Spec_Ops_Env
        trainer = PPO(config=config, env=Spec_Ops_Env)

        # You can just do PPO or DQN but we wanted to show how to customize
        #from ray.rllib.agents.ppo import PPOTrainer
        #trainer = PPOTrainer(config, env=YourEnvironment)

        trainer.train()