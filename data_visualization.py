# The steps of visualizing the data before and after the data analysis.
from keras.models import load_model
m = load_model('my_model.keras')

# enc_dec_model.save("my_model.keras")
loss_values = m.history['loss']
accuracy_values = m.history['accuracy']
mse_values = m.history['mse']

# Print the loss and metric values for each epoch
for epoch in range(len(loss_values)):
    print(f"Epoch {epoch+1}: Loss = {loss_values[epoch]}, Accuracy = {accuracy_values[epoch]}, MSE = {mse_values[epoch]}")

