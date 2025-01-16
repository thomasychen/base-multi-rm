<img src="LOTaD.png" alt="LOTad Logo" width="100" />

# LOTaD: Learning Optimal Task Decompositions for Multiagent Reinforcement Learning

## Installation Instructions

1. Ensure you have Python 3.11 with:
   ```sh
   python --version
   ```

2. (Optional) Set up a Python environment and activate it:
   ```sh
   python -m venv env
   source env/bin/activate  # On Windows use `env\Scripts\activate`
   ```

3. In the repository, install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

## Environments Overview

- **Repairs (Motivating Example)**: A team of three agents must visit a headquarters (HQ) and then visit two communication stations in any order to make repairs. Agents must navigate around a hazardous region that prevents more than one agent from entering at a time.
- **Cooperative Buttons**: Agents must press a series of buttons in a particular order to reach a goal location. Traversing certain regions is only possible once the corresponding button has been pressed.
- **Four-Buttons**: Two agents must press four buttons (yellow, green, blue, red) in an environment, with an ordering constraint that the yellow button must be pressed before the red button.
- **Cramped-Corridor**: Two agents must navigate a small corridor to reach the pot at the end while avoiding collisions, then deliver the soup.
- **Asymmetric-Advantages**: Two agents are in separate rooms, each with access to a different set of resources, and must coordinate to deliver a soup.

### Environment Name Translations
- **Four-Buttons**: `buttons_challenge`
- **Cooperative Buttons**: `easy_buttons`
- **Repairs**: `motivating_example`
- **Asymmetric Advantages**: `custom_island`
- **Cramped Corridor**: `interesting_cramped_room`

## Example Command

```sh
python run.py --assignment_methods UCB --num_iterations 5 \
  --wandb t --decomposition_file mono_interesting_cramped_room.txt \
  --experiment_name interesting_cramped_room --is_monolithic f \
  --env overcooked --render f --video f \
  --add_mono_file mono_interesting_cramped_room.txt --num_candidates 10 \
  --timesteps 1000000
```

### Important Parameters for Trials

- `--add_mono_file`: Remove this parameter if you don't want to add the monolithic embedding.
- `--experiment_name` and decomposition file names: Change them using `interesting_cramped_room`, `custom_island`, `easy_buttons`, `buttons_challenge`, or `motivating_example`.
- `--env`: 
  - Use `overcooked` for **Cramped-Corridor** and **Asymmetric-Advantages**.
  - Use `buttons` for **Four-Buttons**, **Cooperative Buttons**, and **Repairs**.
- `--num_candidates`: The number of decompositions to consider. Setting it to 1 means using the top decomposition by the ATAD definition.
- `--render`: Shows each timestep during execution.
- `--video`: Saves videos of evaluation episodes.
- `--timesteps`: Total training time per iteration.

If you change the decomposition file to `individual_{exp_name}` and do not pass `--num_candidates`, you can run the task training each agent on just the monolithic reward machine.

---

## Adding a New Environment

If you want to add a new environment (that is **neither** a button-based environment nor Overcooked-based), you will need the following files/classes. These ensure your environment is integrated correctly into LOTaD’s multi-agent reward machine structure.

1. **MDP Definition**  
   - **File**: `mdps/(new_env).py` **OR** an importable MDP from an online repo.  
   - **Description**: This file contains the original environment logic. You can also wrap an existing environment as needed.  
   - **Example**: `mdps/easy_buttons_env.py` or `jaxmarl.environments.overcooked`.

2. **MDP Label Wrapper**  
   - **File**: `mdp_label_wrappers/(env_layout_name)_labeled.py`  
   - **Description**: A labeler that emits “events” relevant to the reward machine transitions. **Must inherit** from the abstract class in `mdp_label_wrappers/generic_mdp_labeled.py`.  
   - **Example**: `mdp_label_wrappers/overcooked_interesting_cramped_room_labeled.py` or `mdp_label_wrappers/easy_buttons_mdp_labeled.py`.

3. **PettingZoo Product Environment**  
   - **File**: `pettingzoo_product_env/(new_env)_product_env.py`  
   - **Description**: Implements an interface matching the PettingZoo `ParallelEnv` style (plus some extra functionality needed by our repo). This class is directly used by the MARL algorithm. **Must inherit** from `pettingzoo_product_env/pettingzoo_product_env.py`.  
   - **Example**: `pettingzoo_product_env/overcooked_product_env.py`.

4. **Config File**  
   - **File**: `config/(new_env)/(env_layout_name).yaml`  
   - **Description**: Contains configuration for the environment layout and algorithm hyperparameters. The main requirement is that `labeled_mdp_class` matches exactly the class name of the labeler above.  
   - **Example**: `config/overcooked/interesting_cramped_room.yaml`.

5. **Reward Machines**  
   - **File**: `reward_machines/(new_env)/(env_layout_name)/{mono_* || individual_*}.txt`  
   - **Description**: Simple text-based edge definitions for the reward machines.  
     - `mono_*` defines the **group task** for this layout.  
     - `individual_*` defines the same reward machine replicated `NUM_AGENTS` times and connected to an initial auxiliary node 0 (used for baseline comparisons).  
   - **Example**: `reward_machines/overcooked/interesting_cramped_room/*`.

### Choosing an MDP Foundation

When writing your new environment, you can:
- Use a step function that takes in an `agent: action` dictionary (standard multi-agent format).
- Or have a single-agent step function but manually synchronize among agents.

**Examples**:
- [JaxMARL Overcooked Env](https://github.com/FLAIROx/JaxMARL/tree/main/jaxmarl/environments/overcooked)
- If you prefer a simpler environment or want to design your own, check out `base-multi-rm/mdps/` for how the button environments are set up.

---

## Adapting Overcooked to LOTaD

Below is an overview of how the **Cramped-Corridor** environment from Overcooked was adapted. You can use this as a reference for **any** new environment:

1. **`pettingzoo_product_env/overcooked_product_env.py`**  
   - This class implements the interface inherited from `pettingzoo_product_env/pettingzoo_product_env.py`.  
   - **Key methods**:
     - **`__init__`**:  
       - `manager`: used to execute UCB selection on reward machine decompositions.  
       - `labeled_mdp_class`: which layout’s MDP labeler to use.  
       - `reward_machine`: which reward machine to load.  
       - `config`: various config variables.  
       - `max_agents`: number of agents.  
       - `test`: whether env is in test mode.  
       - `is_monolithic`: (deprecated).  
       - `Addl_mono_rm`: whether to add a global RM (if using group state knowledge).  
       - `Render_mode`: whether to visualize the environment.  
       - `Monolithic_weight`: how much weight the group task reward has vs. decomposed task rewards.  
       - `Log_dir`: where to store logs.  
       - `Video`: whether to log video.  
       - Overcooked uses a `reset_key` for randomization.  
     - **`observation_space`** and **`action_space`**: Derived from the underlying MDP.  
     - **`reset`**: Resets the MDP and the reward machine state. Combines them into a single observation (via `flatten_and_add_rm`).  
     - **`step`**:  
       1. Forward the agents’ actions to the MDP.  
       2. Obtain labels from the labeled MDP.  
       3. For each label, update the reward machines if terminal states are reached.  
       4. Handle terminations (`terminations`, `truncations`) and return `obs`, `rewards`, `infos`, etc.  
     - **`render`**: Uses the `OvercookedVisualizer` if available. (For a custom environment, see how `buttons_product_env` does animations from scratch.)  
     - **`send_animation`**: Used to send visualizations (e.g., to W&B).

2. **`mdp_label_wrappers/overcooked_interesting_cramped_room_labeled.py`**  
   - Imports the specific Overcooked layout.  
   - **`get_mdp_label(state)`**: Outputs a string label (event) based on the current state. This label is looked up by the reward machine.  

3. **Reward Machines** (e.g., `reward_machines/overcooked/interesting_cramped_room/`)  
   - **`mono_interesting_cramped_room.txt`**: Contains the global reward machine transitions for this layout.  
   - **`individual_interesting_cramped_room.txt`**: Baseline decomposition with one reward machine per agent.

4. **Config**: `config/overcooked/interesting_cramped_room.yaml`  
   - Must specify `labeled_mdp_class: OvercookedInterestingCrampedRoomLabeled` (for example).  
   - Contains all hyperparameters for training.

5. **`run.py`**  
   - At line ~279, we load the training env:
     ```python
     if args.env == "overcooked":
         env = OvercookedProductEnv(**train_kwargs)
     ```
   - Similarly at line ~295 for test env.  
   - **Example**:
     ```sh
     python run.py --assignment_methods UCB --num_iterations 1 --wandb t \
       --decomposition_file mono_interesting_cramped_room.txt \
       --experiment_name interesting_cramped_room --is_monolithic f \
       --env overcooked --render f --video f \
       --add_mono_file mono_interesting_cramped_room.txt --num_candidates 10 \
       --timesteps 1000000
     ```

### Plotting & Logging
If you want to log plots:
```sh
python make_plot.py --log_folder base-multi-rm/Final_logs/motivating_example/mono_ablate \
                    --ci 0.95 --sm 5 --g 0.99
```
- `mono_ablate` might contain:
  ```
  LOTaD
    ├─ Iteration_1_seed_1
    └─ Iteration_1_seed_2
  LOTaD (No Overall)
    ├─ Iteration_1_seed_1
    └─ Iteration_1_seed_2
  ```
- The script averages seeds to produce two lines on the resulting plot (LOTaD vs. LOTaD (No Overall)).

---

**Happy experimenting with LOTaD!** For any questions or issues, please open an issue or pull request.
