import numpy as np

inputs = [1, 4, 3, 2, 1]
hidden_layer_size = 7
output_layer_size = 3
label = [0, 1, 0]
learning_rate = 0.03

def init_weights(inputs_size, layer_size):
    weights = np.random.uniform(-1, 1, (layer_size, inputs_size))
    return weights

def dot_product(weights, inputs):
    return np.dot(weights, inputs)

def activation_function(x):
    return 1 / (1 + np.exp(-x))

def output_function(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def network_error(label, output):
    return label - output

def delta_output(output, error):
    return error * (1 - output**2)

def delta_hidden(delta_o, output, weights):
    return np.dot(np.transpose(weights), delta_o) * (1 - output**2)

def weight_adjust(weights, change, learning_rate):
    half = (learning_rate * np.transpose(change))
    weights = weights + half
    return weights

# Feedforward
weights1 = init_weights(len(inputs), hidden_layer_size)
hidden_layer_output = activation_function(dot_product(weights1, inputs))
weights2 = init_weights(hidden_layer_size, output_layer_size)
final_output = output_function(dot_product(weights2, hidden_layer_output))

# Backpropagation
error = network_error(label, final_output)
delta_o = delta_output(final_output, error)
change_output = np.outer(hidden_layer_output, delta_o)
weights2 = weight_adjust(weights2, change_output, learning_rate)
delta_h = delta_hidden(delta_o, hidden_layer_output, weights2)
change_hidden = np.outer(inputs, delta_h)
weights1 = weight_adjust(weights1, change_hidden, learning_rate)

# Print results
print("Input: ", inputs)
print("Target output: ", label)
print("Final output: ", final_output)
print("Error: ", error)
print("Weights in hidden layer: ", weights1)
print("Weights in output layer: ", weights2)