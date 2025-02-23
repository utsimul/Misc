#LEARNING PROJECT -> aim: to learn basics of GNN (GCN)
#this code attempts to classify whether a particular node belongs to community 
#0 or 1 out of the 2 communities based on the set of nodes, edges and node info.

import numpy as np

N = 34
A = np.zeros((N,N))

# Define edges based on Zachary's Karate Club graph
edges = [
    (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 10), (0, 11), (0, 12),
    (0, 13), (0, 17), (0, 19), (0, 21), (0, 31), (1, 2), (1, 3), (1, 7), (1, 13), (1, 17),
    (1, 19), (1, 21), (1, 30), (2, 3), (2, 7), (2, 27), (2, 28), (2, 32), (3, 7), (3, 12),
    (3, 13), (4, 6), (4, 10), (5, 6), (5, 10), (6, 16), (8, 30), (8, 32), (8, 33), (9, 33),
    (13, 33), (14, 32), (14, 33), (15, 32), (15, 33), (18, 32), (18, 33), (19, 33), (20, 32),
    (20, 33), (22, 32), (22, 33), (23, 25), (23, 27), (23, 29), (23, 32), (23, 33), (24, 25),
    (24, 27), (24, 31), (25, 31), (26, 29), (26, 33), (27, 33), (28, 31), (28, 33), (29, 33),
    (30, 32), (30, 33), (31, 32), (31, 33), (32, 33)
]

for i,j in edges:
    A[i,j] = 1
    A[j,i] = 1 #undirected graph

I = np.eye(N) #identity matrix of order N
A_hat = A + I
D_hat = np.diag(np.sum(A_hat,axis=1))

D_hat_pow_min1by2 = np.linalg.inv(np.sqrt(D_hat))
A_normalized = D_hat_pow_min1by2 @ A_hat @ D_hat_pow_min1by2

#Node features (One Hot encoding)
H = I

def gcn_layer(A, H, W):
    return np.maximum(0,A @ H @ W)

input_dim = N
hidden_dim = 16
output_dim = 2

np.random.seed(42)
W0 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2. / input_dim)
W1 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2. / hidden_dim)


def softmax(X):
    exp_X = np.exp(X - np.max(X,axis=1, keepdims=True))
    return exp_X / np.sum(exp_X, axis=1, keepdims=True)

def cross_entropy(predictions,labels):
    N = predictions.shape[0]
    return -np.sum(labels* np.log(predictions + 1e-9)) / N #adding 10^-9 ensures no log(0) case!

def accuracy(predictions,labels):
    pred_classes = np.argmax(predictions,axis=1)
    true_classes = np.argmax(labels,axis=1)
    return np.mean(pred_classes==true_classes)

y_true = np.zeros((N, output_dim))
y_true[:17, 0] = 1  # Class 0
y_true[17:, 1] = 1  # Class 1


learning_rate = 0.1
epochs = 200

for epoch in range(epochs):
    H1 = gcn_layer(A_normalized,H,W0) #first GCN layer
    H2 = A_normalized @ H1 @ W1 #second layer (no RELU)
    output = softmax(H2)

    loss = cross_entropy(output,y_true)
    acc = accuracy(output, y_true)

    grad_otpt = (output - y_true)/N #this is also the derivative of loss fn
    grad_W1 = H1.T @ A_normalized.T @ grad_otpt
    grad_H1 = grad_otpt @ W1.T
    grad_H1 = grad_H1 * (H1 > 0)  # ReLU derivative
    grad_W0 = H.T @ A_normalized.T @ grad_H1
    print("Grad W1 norm:", np.linalg.norm(grad_W1))
    print("Grad W0 norm:", np.linalg.norm(grad_W0))


    W1 -= learning_rate * grad_W1
    W0 -= learning_rate * grad_W0

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {acc:.4f}")

H1 = gcn_layer(A_normalized, H, W0)
H2 = np.maximum(0, A_normalized @ H1 @ W1)  # Apply ReLU
output = softmax(H2)
final_acc = accuracy(output, y_true)
print(f"Final accuracy: {final_acc:.4f}")
print("First 5 node predictions: (probabilities) \n", output[:5])
