import numpy as np

# Input features (x1, x2)
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Target output for AND gate
y = np.array([0, 0, 0, 1])

# ----------------------------------------------
# Step 3: Initialize weights and bias randomly
# ----------------------------------------------
np.random.seed(42)            # To make results reproducible
weights = np.random.rand(2)   # Two weights for two inputs
bias = np.random.rand(1)      # One bias term
learning_rate = 0.1           # Step size for weight updates

print("Initial weights:", weights)
print("Initial bias:", bias)
print("----------------------------------")

# ----------------------------------------------
# Step 4: Define Activation Function (Step Function)
# ----------------------------------------------
def step_function(x):
    """Returns 1 if input >= 0, else 0"""
    return 1 if x >= 0 else 0

# ----------------------------------------------
# Step 5: Training the Perceptron
# ----------------------------------------------
epochs = 10   # Number of iterations over the dataset

for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}")
    for i in range(len(X)):
        #  Calculate the weighted sum
        linear_output = np.dot(X[i], weights) + bias

        #  Apply activation function
        predicted = step_function(linear_output)

        #  Calculate error (target - predicted)
        error = y[i] - predicted

        #  Update weights and bias
        weights = weights + learning_rate * error * X[i]
        bias = bias + learning_rate * error

        # Display progress for students
        print(f"Input: {X[i]}, Target: {y[i]}, Predicted: {predicted}")
        print(f"Updated weights: {weights}, Updated bias: {bias}")

print("\nTraining complete!")
print("----------------------------------")
print("Final weights:", weights)
print("Final bias:", bias)

# ----------------------------------------------
# Step 6: Testing the Trained Model
# ----------------------------------------------
print("\n--- Testing the Trained Perceptron ---")
for i in range(len(X)):
    linear_output = np.dot(X[i], weights) + bias
    predicted = step_function(linear_output)
    print(f"Input: {X[i]} --> Predicted Output: {predicted}")



