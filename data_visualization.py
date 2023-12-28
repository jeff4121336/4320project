# Student ID: 1155158477
# Name: YAU YUK TUNG

# The steps of training the model based on the preprocessed data, and the steps of assessing the performance
# metrics of the trained model.
import numpy as np
import matplotlib.pyplot as plt
from data_analysis import loss_values, accuracy_values, mse_values

epochs = np.arange(1, len(loss_values) + 1).tolist()
fig, axs = plt.subplots(3, 1, figsize=(8, 12))

axs[0].plot(epochs, loss_values)
axs[0].set_ylabel('Loss')
axs[0].set_xlabel('Epochs')

axs[1].plot(epochs, accuracy_values)
axs[1].set_ylabel('Accuracy')
axs[1].set_xlabel('Epochs')

axs[2].plot(epochs, mse_values)
axs[2].set_ylabel('MSE')
axs[2].set_xlabel('Epochs')

plt.savefig('model_data.png', bbox_inches='tight')

print("Data visualization completed.")
print("--END--")