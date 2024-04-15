import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Provided data
data = [
    [2590, 0, 0, 0],
    [0, 2172, 430, 0],
    [0, 754, 1633, 0],
    [0, 197, 0, 0]
]

# Convert the data to a NumPy array
data_array = np.array(data)

# Create a heatmap
sns.heatmap(data_array, annot=True, fmt='.1f', cmap='viridis')

# Show the plot
plt.show()
