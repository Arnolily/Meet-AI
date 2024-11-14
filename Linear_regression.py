import numpy as np
import matplotlib.pyplot as plt
import torch


# Define the function
def linear_function(x):
    return 2 * x + 1

# Generate random x values
x_values = np.linspace(0, 10, 1000)

# Calculate corresponding y values with added noise
noise = np.random.normal(0, 1, x_values.shape)
y_values = linear_function(x_values) + noise

# Store the generated data points in the data list
data = list(zip(x_values, y_values))


net = torch.nn.Sequential(
    torch.nn.Linear(1, 1)
)

for x, y in zip(x_values, y_values):
    x_tensor = torch.tensor([x], dtype=torch.float32)
    y_tensor = torch.tensor([y], dtype=torch.float32)

    # Forward pass
    y_pred = net(x_tensor)

    # Compute the loss
    loss = torch.nn.functional.mse_loss(y_pred, y_tensor)

    # Zero the gradients
    net.zero_grad()

    # Backward pass
    loss.backward()

    # Update the weights
    for param in net.parameters():
        param.data = param.data - 0.001 * param.grad
        
weights = list(net.parameters())[0].data.numpy()
bias = list(net.parameters())[1].data.numpy()

# Print the linear equation
print(f"Predicted linear function: y = {weights[0]} * x + {bias[0]}")

# Plot the data points
plt.scatter(x_values, y_values, label='Data points')
plt.plot(x_values, linear_function(x_values), color='red', label='True function')
plt.plot(x_values, [net(torch.tensor([x], dtype=torch.float32)).item() for x in x_values], color='green', label='Predicted function')
plt.legend()
plt.show()