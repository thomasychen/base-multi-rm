from utils.plot_utils import generate_plots, generate_plots_legacy
import numpy as np

generate_plots_legacy('logs/20240525-210407', 1)

# import numpy as np

# # Load the npz file
# data = np.load('/Users/nikhil/Desktop/research_rl/base-multi-rm/logs/20240525-210407/add/iteration_1/evaluations.npz')

# # List all items in the file
# print("Keys in the npz file:", data.files)

# # Access data
# for key in data.files:
#     print(f"Data under {key}:")
#     print(data[key])

# # Optionally close the file, typically not necessary
# # data.close()

