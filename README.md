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
   source env/bin/activate # On Windows use `env\Scripts\activate`
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
- **Four-Buttons**: buttons_challenge
- **Cooperative Buttons**: easy_buttons
- **Repairs**: motivating_example
- **Asymmetric Advantages**: custom_island
- **Cramped Corridor**: interesting_cramped_room

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
- `--experiment_name` and decomposition file names: Change them using `interesting_cramped_room`, `custom_island`, `easy_buttons`, `buttons_challenge`, `motivating_example`.
- `--env`: Use `overcooked` for Cramped-Corridor and Asymmetric-Advantages, `buttons` for Four-Buttons, Cooperative Buttons, and Repairs.
- `--num_candidates`: The number of decompositions to consider. Setting it to 1 means using the top decomposition by ATAD definition.
- `--render`: Shows each timestep during execution.
- `--video`: Saves videos of evaluation episodes.
- `--timesteps`: Total training time per iteration.

If you change the decomposition file to `individual_{exp_name}` and do not pass `--num_candidates`, you can run the task training each agent on just the monolithic reward machine.