from ray.tune.registry import register_env
import ray
# import the pettingzoo environment
from pettingzoo.butterfly import prison_v3
# import rllib pettingzoo interface
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env import PettingZooEnv
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from train import Spec_Ops_Env
# define how to make the environment. This way takes an optional environment config, num_floors
env_creator = lambda config: prison_v3.env(num_floors=config.get("num_floors", 4))
# register that way to make the environment under an rllib name
register_env('prison', lambda config: PettingZooEnv(env_creator(config)))
# now you can use `prison` as an environment
# you can pass arguments to the environment creator with the env_config option in the config

if __name__=="__main__":
    ray.init()
    env_name='Spec_Ops'
    register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator(config)))

    config = (
        PPOConfig()
        .environment(env=env_name, clip_actions=True)
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
        .debugging(log_level="ERROR")
        .framework(framework="torch")
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    )
    ray.tune.run(
        "PPO",
        name="PPO",
        stop={"timesteps_total": 5000000},
        checkpoint_freq=10,
        local_dir="~/ray_results/" + env_name,
        config=config.to_dict(),
    )
