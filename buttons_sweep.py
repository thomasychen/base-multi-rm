import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import DQN, PPO, SAC, DDPG, HER, HerReplayBuffer
from pettingzoo_product_env.custom_environment.env.custom_environment import MultiAgentEnvironment
import supersuit as ss
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
import yaml
from manager.manager import Manager

def train():
    # config = {
    # "policy_type": "MlpPolicy",
    # # "total_timesteps": 1000000,
    # "env_name": "Buttons",
    # }

    run = wandb.init(project = experiment, entity="reinforce-learn", sync_tensorboard=True)
    config = wandb.config
    with open('config/buttons.yaml', 'r') as file:
        buttons_config = yaml.safe_load(file)

        
    # mdp_labeled = HardButtonsLabeled(buttons_config)

    num_agents = 3
    manager = Manager(num_agents=num_agents, assignment_method="multiply")
    env = MultiAgentEnvironment(manager=manager)

    # env.set_manager(manager)
    # env.reset()


    print(f"Starting training on {str(env.metadata['name'])}.")



    env = ss.black_death_v3(env)

    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class="stable_baselines3")
    env = VecMonitor(env)

    model = DQN(
        "MlpPolicy",
        env,
        verbose=3,
        # action_noise=action_noise,
        exploration_initial_eps=1,
        exploration_final_eps=config.exploration_final_eps, 
        exploration_fraction=config.exploration_fraction,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        # target_update_interval=100,
        gamma = config.gamma,
        buffer_size=config.buffer_size,
        target_update_interval=config.target_update_interval,
        tensorboard_log=f"runs/{run.id}"
    )

    manager.set_model(model)
    env.reset()


    model.learn(total_timesteps=2000000, callback=WandbCallback(
        verbose=2,
    ))


if __name__ == "__main__": 
    experiment = "test_pettingzoo_sb3_sweep"

    sweep_configuration = {
        "method": "bayes",
        "metric": {"goal": "maximize", "name": "rollout/ep_rew_mean"},
        "parameters": {
            "exploration_final_eps": {'distribution': 'uniform', 'min': 0.05, 'max': 0.2},
            "exploration_fraction": {"distribution": "uniform", "min": 0.1, "max": 0.3},
            "batch_size": {"values": [128, 256, 512, 1024, 2048, 4096]},
            "learning_rate": {'distribution': 'uniform', 'min': 1e-6, 'max': 1e-2},
            "gamma": {"values": [0.75, 0.8, 0.85, 0.875, 0.9]},
            "buffer_size": {"values": [5000, 10000, 20000]},
            "target_update_interval": {"values": [1000, 5000, 10000]},
        },
    }

    sweep_id = wandb.sweep(sweep=sweep_configuration, project=experiment, entity="reinforce-learn")
    wandb.agent(sweep_id, function=train)