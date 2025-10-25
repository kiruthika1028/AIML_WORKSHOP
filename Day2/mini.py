import numpy as np

np.random.seed(42)
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

X = np.vstack([X]*100)  
y = np.vstack([y]*100)

perm = np.random.permutation(len(X))
X, y = X[perm], y[perm]

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

input_neurons = 2
hidden_neurons = 4
output_neurons = 1

w1 = np.random.rand(input_neurons, hidden_neurons)
b1 = np.random.rand(1, hidden_neurons)
w2 = np.random.rand(hidden_neurons, output_neurons)
b2 = np.random.rand(1, output_neurons)

learning_rate = 0.1
epochs = 5000

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)


for epoch in range(epochs):
    
    hidden_input = np.dot(X_train, w1) + b1
    hidden_output = sigmoid(hidden_input)
    output_input = np.dot(hidden_output, w2) + b2
    predicted = sigmoid(output_input)

   
    error = y_train - predicted
    d_output = error * sigmoid_derivative(predicted)
    d_hidden = d_output.dot(w2.T) * sigmoid_derivative(hidden_output)


    w2 += hidden_output.T.dot(d_output) * learning_rate
    b2 += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    w1 += X_train.T.dot(d_hidden) * learning_rate
    b1 += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

    if (epoch+1) % 1000 == 0:
        loss = np.mean(np.square(error))
        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}")


hidden = sigmoid(np.dot(X_test, w1) + b1)
out = sigmoid(np.dot(hidden, w2) + b2)
pred_labels = (out > 0.5).astype(int)

print("\nStudent Pass/Fail Predictions:")
correct = 0
for i in range(len(X_test)):
    print(f"Input: {X_test[i]} -> Predicted: {pred_labels[i,0]} | Actual: {y_test[i,0]}")
    if pred_labels[i,0] == y_test[i,0]:
        correct += 1


accuracy = correct / len(X_test)
print(f"\nOverall Accuracy: {accuracy*100:.2f}%")

