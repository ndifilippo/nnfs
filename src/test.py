#%% 
import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from dense_layer import Layer_Dense
from activation_function import Activation_ReLU, Activation_Softmax
from loss_function import Loss, Loss_CategoricalCrossentropy

#%% 
# Create dataset
X, y = spiral_data(samples=100, classes=3)

#%% 
# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)
#%% 
# Create ReLU activation (to be used with Dense layer):
activation1 = Activation_ReLU()

#%% 
# Create second Dense layer with 3 input features (as we take output
# of previous layer here) and 3 output values
dense2 = Layer_Dense(3, 3)

#%% 
# Create Softmax activation (to be used with Dense layer):
activation2 = Activation_Softmax()

#%% 
# Create a loss function 
loss_function = Loss_CategoricalCrossentropy()




#%%
# Helper Variables 
lowest_loss = 9999999
best_dense1_weights = dense1.weights.copy()
bets_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
bets_dense2_biases = dense2.biases.copy()

for iteration in range(1000):

    #Update weights with some small random variable
    dense1.weights += 0.05 * np.random.randn(2,3)
    dense1.biases +=  0.05 * np.random.randn(1,3)
    dense2.weights += 0.05 * np.random.randn(3,3)
    dense2.biases +=  0.05 * np.random.randn(1,3)

    # Make a forward pass of our training data through this layer
    dense1.forward(X)

    # Make a forward pass through activation function
    # it takes the output of first dense layer here
    activation1.forward(dense1.output)

    # Make a forward pass through second Dense layer
    # it takes outputs of activation function of first layer as inputs
    dense2.forward(activation1.output)

    # Make a forward pass through activation function
    # it takes the output of second dense layer here
    activation2.forward(dense2.output)

    loss = loss_function.calculate(activation2.output, y)

    # Calculate accuracy from output of activation2 and targets
    # calculate values along first axis
    predictions = np.argmax(activation2.output, axis=1)
    accuracy = np.mean(predictions == y)
    # If loss is smaller - print and save weights and biases aside
    if loss < lowest_loss:
        print('New set of weights found, iteration:', iteration,'loss:', loss, 'acc:', accuracy)
        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.biases.copy()
        lowest_loss = loss
    # Revert weights and biases
    else:
        dense1.weights = best_dense1_weights.copy()
        dense1.biases = best_dense1_biases.copy()
        dense2.weights = best_dense2_weights.copy()
        dense2.biases = best_dense2_biases.copy()
