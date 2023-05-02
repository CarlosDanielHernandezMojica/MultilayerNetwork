import numpy as np

def init_weights(inputs_size, layer_size):
    weights = np.random.uniform(-1, 1, (layer_size, inputs_size))
    return weights

def dot_product(weights, inputs):
    return np.dot(weights, inputs)

def activation_function(dot_products):
    outputs = 1/(1 + np.exp(-(dot_products)))
    return outputs

def delta_output(output, error):
    delta_o = ((1 - (output)**2)) * error
    return delta_o

def change_variable(delta_o, outputs):
    change = np.outer(delta_o, outputs)
    return change

def weight_adjust(weights, change, learning_rate):
    weights = weights + learning_rate * change
    return weights

def network_error(label, output):
    return label - output

def hidden_layer_error(delta_o, hidden_weights):
    dot = np.dot(hidden_weights.T, delta_o)
    return dot

inputs = [1, 4, 3, 2, 1]
hidden_layer_size = 7
output_layer_size = 3
label = [0, 1, 0]
learning_rate = 0.03

# Initialize weights
weights1 = init_weights(len(inputs), hidden_layer_size)
weights2 = init_weights(hidden_layer_size, output_layer_size)

# Forward pass
dot_hidden_layer = dot_product(weights1, inputs)
output_hidden = activation_function(dot_hidden_layer)

dot_output_layer = dot_product(weights2, output_hidden)
final_output = activation_function(dot_output_layer)

# Backward pass
error = network_error(label, final_output)
delta_o = delta_output(final_output, error)

change = change_variable(delta_o, output_hidden)
weights2 = weight_adjust(weights2, change, learning_rate)

hidden_error = hidden_layer_error(delta_o, weights2)
delta_o_hidden = delta_output(output_hidden, hidden_error)

change_hidden = change_variable(delta_o_hidden, inputs)
weights1 = weight_adjust(weights1, change_hidden, learning_rate)


print("Pesos ajustados de la capa de salida:")
print(weights2)

print("Salida final después de ajustar los pesos:")
print(final_output)