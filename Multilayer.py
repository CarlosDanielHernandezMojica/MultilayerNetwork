import numpy as np
## Doubts
## Is it necessary a bias?
## How do we calculate the change variable
## How do we adjust the weights
## Is the (dsig(Xoi)) in the delta output the actual output of the neuron

inputs = [1, 4, 3, 2, 1]

hidden_layer_size = 7
output_layer_size = 3

label = [0, 1, 0]

learning_rate = 0.03

#Remove the round method
def init_weights(inputs_size, layer_size):
    weights = np.empty([layer_size, inputs_size])
    
    for i in range(layer_size):
        for j in range(inputs_size):
            weight = np.random.uniform(-1, 1)
            weights[i][j] = np.round(weight, 2)
    
    return weights

def dot_product(weights, inputs):
    return np.dot(weights, inputs)

def activation_function(dot_products):
    outputs = np.empty(len(dot_products))
    
    for i in range(len(dot_products)):
        outputs[i] = 1/(1 + np.exp(-(dot_products[i])))
    return outputs

def network_error(label, output):
    return label - output

def hidden_layer_error(delta_o, hidden_weights):
    tranposed = np.transpose(hidden_weights)
    dot = np.dot(tranposed, delta_o)
    return dot
    
def delta_output(output, error):
    delta_o = ((1 - (output)**2)) * error 
    print("Delta o")
    print(delta_o)
    
    return delta_o

## This one is right, so next thing is do the same with the hidden layer
def change_variable(delta_o, outputs):
    change = np.zeros([len(delta_o), len(outputs[0])])
    print("empty change")
    print(change)

    for i in range(len(delta_o)):
        for j in range(len(outputs[0])):
            change[i][j] = delta_o[i] * outputs[i][j]

    print("change")
    print(np.round(change))
    return np.round(change, 3)


def weight_adjust(weights, change, learning_rate):
    
    half = (learning_rate * change)
    print("learnig_rate * change")
    print(half)
    weights = weights + half
    print("weights adjusted")
    print(weights)
    return weights

print("Inputs")
print(inputs)

weights1 = init_weights(len(inputs), hidden_layer_size)
print("hidden_layer_weigths")
print(weights1)

dot_hidden_layer = dot_product(weights1, inputs)
print("dot_hidden_layer")
print(dot_hidden_layer)

output_hidden = activation_function(dot_hidden_layer)
print("output_hidden")
print(output_hidden)

weights2 = init_weights(len(weights1), output_layer_size)
print("output_layer_weights")
print(weights2)

dot_output_layer = dot_product(weights2, output_hidden)
print("dot_output_layer")
print(dot_output_layer)

final_output = activation_function(dot_output_layer)
print("final_output")
print(final_output)

error = network_error(label, final_output)
print("network_error")
print(error)

## Revisar desde aqui
before_dot = np.round(np.multiply(weights2, output_hidden), 3)
print("before dot")
print(before_dot)
print("delta output")
delta_o = delta_output(final_output, error)

change = change_variable(delta_o, before_dot)

print("weight adjust")
weight_adjust(weights2, change, learning_rate)

print("hidden error")
hidden_error = hidden_layer_error(delta_o, weights2)
print(hidden_error)

print("hidden error")
delta_o_hidden = delta_output(output_hidden, hidden_error)
print(delta_o_hidden)

change_hidden = change_variable(delta_o_hidden, inputs)













def test():
# def hidden_layer(matrix):
#     hidden_weights = []

#     for i in range(len(matrix[0])):
#         hidden_weights.append(list(np.random.uniform(-1, 1, 3)))

#     print("Hidden weights")
#     print("-"*20)
#     print(hidden_weights)
#     return hidden_weights


# def calculate_z(xi, wi):
    
#     z = np.sum(np.multiply(xi,wi))
    
#     return z

# def activation(z):
#     return -1 if z < 0 else 1

# xi = [3 , 2, 4]
# wi = [2 , 3, 2]

    pass
