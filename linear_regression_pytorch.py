import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. Create and prepare dataset
X_numpy, Y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)

# Normalize the data
scaler = StandardScaler()
X_numpy = scaler.fit_transform(X_numpy)

# Convert to tensors
X = torch.tensor(X_numpy, dtype=torch.float32)
y = torch.tensor(Y_numpy, dtype=torch.float32).view(-1, 1)

# 2. Model, loss function, and optimizer
model = nn.Linear(X.shape[1], 1)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)  # Reduced learning rate

# 3. Training loop
epochs = 1000

for epoch in range(epochs):
    y_pred = model(X)
    loss = criterion(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# 4. Plotting
with torch.no_grad():
    predicted = model(X).numpy()

plt.plot(X_numpy, Y_numpy, 'ro', label='Original data')
plt.plot(X_numpy, predicted, 'b', label='Fitted line')
plt.legend()
plt.show()
