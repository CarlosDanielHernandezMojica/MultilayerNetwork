import numpy as np

inputs = [1, 4, 3, 2, 1]
hidden_layer_size = 7
output_layer_size = 3
label = [0, 1, 0]
learning_rate = 0.03

def init_weights(inputs_size, layer_size):
    weights = np.random.uniform(-1, 1, [layer_size, inputs_size])
    return weights

def dot_product(weights, inputs):
    return np.dot(weights, inputs)

def activation_function(dot_products):
    return 1 / (1 + np.exp(-dot_products))

def derivative_activation_function(output):
    return output * (1 - output)

def network_error(label, output):
    return label - output

def delta_output(output, error):
    return error * derivative_activation_function(output)

def hidden_layer_error(delta_o, hidden_weights):
    return np.dot(hidden_weights.T, delta_o)

def change_variable(delta, output):
    return np.outer(delta, output)

def weight_adjust(weights, change, learning_rate):
    return weights + learning_rate * change

# Forward pass
weights1 = init_weights(len(inputs), hidden_layer_size)
dot_hidden_layer = dot_product(weights1, inputs)
output_hidden = activation_function(dot_hidden_layer)
weights2 = init_weights(len(output_hidden), output_layer_size)
dot_output_layer = dot_product(weights2, output_hidden)
final_output = activation_function(dot_output_layer)

# Backward pass
output_error = network_error(label, final_output)
delta_o = delta_output(final_output, output_error)
change2 = change_variable(delta_o, output_hidden)
weights2 = weight_adjust(weights2, change2, learning_rate)
hidden_error = hidden_layer_error(delta_o, weights2)
delta_h = delta_output(output_hidden, hidden_error)
change1 = change_variable(delta_h, inputs)
weights1 = weight_adjust(weights1, change1, learning_rate)

print("Inputs")
print(inputs)
print("hidden_layer_weigths")
print(weights1)
print("output_layer_weights")
print(weights2)