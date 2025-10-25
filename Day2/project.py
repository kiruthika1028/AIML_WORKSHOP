import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

np.random.seed(30)


X_base = np.array([[0,0],[0,1],[1,0],[1,1]])   
y_base = np.array([[0],[1],[1],[0]])          


def create_noisy_dataset(X, y, n_copies=250, noise_prob=0.05):
    Xs, ys = [], []
    for _ in range(n_copies):
        idx = np.random.choice(len(X))
        sample = X[idx].copy()
       
        for j in range(sample.shape[0]):
            if np.random.rand() < noise_prob:
                sample[j] = 1 - sample[j]
        Xs.append(sample)
        ys.append(y[idx])
    return np.vstack(Xs), np.vstack(ys)

X_noisy, y_noisy = create_noisy_dataset(X_base, y_base, n_copies=500, noise_prob=0.05)


X = np.vstack([X_base, X_noisy])
y = np.vstack([y_base, y_noisy])

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
epochs = 3000

def sigmoid(x):
    return 1/(1 + np.exp(-x))
def sigmoid_derivative(x):
    return x * (1 - x)


for epoch in range(epochs):
    # Forward
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

    if (epoch+1) % 500 == 0:
        loss = np.mean(np.square(y_train - predicted))
        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}")


hidden = sigmoid(np.dot(X_test, w1) + b1)
out = sigmoid(np.dot(hidden, w2) + b2)
pred_labels = (out > 0.5).astype(int)

print("\nConfusion Matrix (NumPy MLP):")
print(confusion_matrix(y_test, pred_labels))
print("\nClassification Report:")
print(classification_report(y_test, pred_labels, digits=4))


print("\nSample predictions:")
for i in range(min(10, len(X_test))):
    print(f"Input: {X_test[i]} -> Pred: {int(pred_labels[i])} (prob={out[i,0]:.3f})  True: {y_test[i,0]}")
