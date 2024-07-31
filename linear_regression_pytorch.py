import torch
import torch.nn as nn
import torch.optim as optim

# Step 1: Prepare the dataset
# Input data (features)
X = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)
# Target data (labels)
Y = torch.tensor([[2.0], [4.0], [6.0], [8.0]], dtype=torch.float32)

n_samples, n_features = X.shape

input_size = n_features
output_size = n_features

# Step 2: Define the Linear Regression model
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        # One input feature and one output feature
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)

# Instantiate the model
model = LinearRegressionModel()

# Step 3: Define the loss function and optimizer
# Mean Squared Error loss
criterion = nn.MSELoss()
# Stochastic Gradient Descent (SGD) optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Step 4: Training loop
# Number of epochs (iterations over the entire dataset)
epochs = 100

for epoch in range(epochs):
    # Forward pass: Compute predicted Y by passing X to the model
    y_pred = model(X)

    # Compute the loss
    loss = criterion(y_pred, Y)

    # Zero gradients, backward pass, and update weights
    optimizer.zero_grad()  # Zero the gradients before backward pass
    loss.backward()  # Backward pass to compute gradients
    optimizer.step()  # Update the weights

    # Print progress every 10 epochs
    if (epoch + 1) % 10 == 0:
        [w,b] = model.parameters()
        print(f'w:{w[0][0].item()} - b:{b[0]}')
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# Step 5: Test the model (inference)
# Make a prediction for a new input value, e.g., X=5
new_input = torch.tensor([[5.0]], dtype=torch.float32)
predicted = model(new_input).item()
print(f'Prediction for input 5: {predicted:.3f}')
