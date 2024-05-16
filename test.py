import numpy as np

data = np.load('./eval_logs/evaluations.npz')

# Print each array in the file
for key in data:
    print(f"Array in '{key}':")
    print(data[key])
    print()  # Adds a blank line for better separation