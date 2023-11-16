from ray import tune
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.a2c import A2C  
from ray.rllib.algorithms.appo import APPO
from custom_environment import Spec_Ops_Env


def train() -> None:
    # config training parameters
    train_config = {
        "env": "spec_ops_env", # MyCustomEnv_v0,
        "framework": "torch",
        "num_workers": 2,
        "num_gpus": 1,  # Add this line to specify using one GPU
        "num_envs_per_worker": 1,
        "model": {
            "fcnet_hiddens": [512, 512, 256],
            "fcnet_activation": "relu",
        },
        "lr": 3e-4,  
        "optimization": {
            "optimizer": "adam",
            "adam_epsilon": 1e-8,
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
        },  
        "gamma": 0.99,
        "num_sgd_iter": 10,  
        "sgd_minibatch_size": 500, 
        "rollout_fragment_length": 500,
        "train_batch_size": 4000,
        "prioritized_replay": True,
        "prioritized_replay_alpha": 0.6,
        "prioritized_replay_beta": 0.4, 
        "buffer_size": 500000,
        "stop": {"episodes_total": 5000000},
        "exploration_config": {},
    }
    stop_criteria = {"training_iteration": 100}

    # start to train
    try:
        results = tune.run(
            PPO, # PPO,
            config=train_config,
            stop=stop_criteria,
            checkpoint_at_end=True,
            checkpoint_freq=50, 
            # restore=model_restore_dir,
            verbose=1,
        )
    except BaseException as e:
        print(f"training error: {e}")
    
if __name__ == "__main__":
    train()
