from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor
import supersuit as ss
from pettingzoo_product_env.overcooked_product_env import OvercookedProductEnv
from jaxmarl import make
from stable_baselines3.common.evaluation import evaluate_policy
from jaxmarl.environments.overcooked import Overcooked, overcooked_layouts

def run_trained_model(model_path, steps):
    # Define environment and configuration
    max_steps = 400
    layout = overcooked_layouts["cramped_room"]
    jax_eval_env = make('overcooked', layout=layout, max_steps=max_steps)
    eval_env = OvercookedProductEnv(jax_eval_env, render_mode = "human", test = True)
    eval_env = ss.black_death_v3(eval_env)
    eval_env = ss.pettingzoo_env_to_vec_env_v1(eval_env)
    eval_env = ss.concat_vec_envs_v1(eval_env, 1, num_cpus=1, base_class="stable_baselines3")
    eval_env = VecMonitor(eval_env)
    

    # Load the trained model
    model = PPO.load(model_path, env=eval_env)
    
    # model.learn(total_timesteps = steps*max_steps)
    # vec_env = model.get_env() 
    # obs = vec_env.reset()
    # obs = eval_env.reset()
    # for i in range(max_steps):
    #     action, _states = model.predict(obs, deterministic=True) 
    #     # print("CURRENT I", i, action)
    #     obs, rewards, dones, info = eval_env.step(action) 
    #     eval_env.render("human")
    # Evaluate the model for the given number of steps
    # mean_reward, std_reward = evaluate_policy(model, eval_env)
    

    print(f"Mean reward: {mean_reward} +/- {std_reward}")

run_trained_model("/Users/thomaschen/base-multi-rm/logs/20240819-230157/best/best_model.zip", 1)