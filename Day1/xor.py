import numpy as np

x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

np.random.seed(30)
input_neurons = 2
hidden_neurons = 2
output_neurons = 1

w1 = np.random.rand(input_neurons, hidden_neurons)
b1 = np.random.rand(1,hidden_neurons)
w2 = np.random.rand(hidden_neurons, output_neurons)
b2 = np.random.rand(1,output_neurons)

learning_rate = 0.1
epochs=10000

def sigmoid(x):
  return 1/(1+np.exp(-x))
def sigmoid_derivative(x):
  return x*(1-x)

for epoch in range(epochs):
  hidden_input = np.dot(x,w1)+b1
  hidden_output = sigmoid(hidden_input)
  output_input = np.dot(hidden_output,w2)+b2
  predicted_output = sigmoid(output_input)

  # Backpropagation
  error = y - predicted_output
  d_output = error * sigmoid_derivative(predicted_output)
  d_hidden = d_output.dot(w2.T) * sigmoid_derivative(hidden_output)

  w2 += hidden_output.T.dot(d_output) * learning_rate
  b2 += np.sum(d_output,axis=0,keepdims=True) * learning_rate
  w1 += x.T.dot(d_hidden) * learning_rate
  b1 += np.sum(d_hidden,axis=0,keepdims=True) * learning_rate

print("\n Testing the Trained MLP for XOR Gate")
for i in range(len(x)):
  hidden_layer_input = np.dot(x[i], w1) + b1
  hidden_layer_output = sigmoid(hidden_layer_input)
  output_layer_input = np.dot(hidden_layer_output, w2) + b2
  predicted = sigmoid(output_layer_input)
  print(f"Input: {x[i]} --> Output: {round(float(predicted))} (Actual Output: {float(predicted):.4f})")