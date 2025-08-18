import numpy as np

class Neuron:
    def __init__(self):
        self.W = np.random.rand(4, 5) * 0.1
        # self.b = 

    def forward(self, x):
        self.input = x
        mid = self.input @ self.W
        self.out = np.tanh(mid)
        return self.out
    
    def backward(self, grad_output):
        grad_mid = grad_output * (1 - self.out**2)
    


# MAIN
x = [[2.0, 3.0, 1.0, 2.5],
     [1.0, -2.5, 0.5, -1.5],
     [-0.5, 3.0, 1.5, 2.0]]

labels = np.array([-1.0, 0.0, 1.0])
labels = labels.reshape(-1, 1)
neuron = Neuron()
print("weights \n", neuron.W)
preds = neuron.forward(x)
print("out after forward \n",  preds)

error = preds - labels
print(f"error \n{error}")

loss = np.mean(error**2)
print(f"final loss \n{loss}")

