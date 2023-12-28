# Student ID: 1155158477
# Name: YAU YUK TUNG
from train_model import results

# enc_dec_model.save("my_model.keras")
loss_values = results.history['loss']
accuracy_values = results.history['accuracy']
mse_values = results.history['mse']

# Print the loss and metric values for each epoch
for epoch in range(len(loss_values)):
    print(f"Epoch {epoch+1}: Loss = {loss_values[epoch]}, Accuracy = {accuracy_values[epoch]}, MSE = {mse_values[epoch]}")

print("Data analysis completed.")