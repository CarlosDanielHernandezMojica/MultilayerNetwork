import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

n_folds = 5
path = "new_iris.csv"
learning_rate = 0.1
hidden_layer_size = 7
output_layer_size = 3
file = open("errors.txt", "a")

def create_folds(n_folds, path):
    df = pd.read_csv(path)
    columns = df.columns
    data = np.array(df)
    np.random.shuffle(data)
    df_shuffled = pd.DataFrame(data, columns=columns)
    
    inputs = df_shuffled.iloc[:, : -1]
    labels = df_shuffled.iloc[:, -1]

    max_values = inputs[:].max()
    min_values = inputs[:].min()
    
    inputs = (inputs[:] - min_values) / (max_values - min_values)
    
    inputs = np.array(inputs)
    labels = np.array(labels)

    folds = np.split(inputs, n_folds)
    label_folds = np.split(labels, n_folds)

    return folds, label_folds

def init_weights(inputs_size, layer_size):
    weights = np.empty([layer_size, inputs_size])
    
    for i in range(layer_size):
        for j in range(inputs_size):
            weight = np.random.uniform(-1, 1)
            weights[i][j] = weight
    
    return weights

def dot_product(weights, inputs):
    return np.dot(weights, inputs)

def activation_function(dot_products):
    outputs = 1 / (1 + np.exp(-dot_products))
    return outputs

def network_error(label, output):
    #n = len(label)
    #error = 1/n * np.sum((np.square(output - label)))
    error = label - output
    #error = np.mean((label - output)**2)
    return error

def hidden_layer_error(delta_o, hidden_weights):
    tranposed = np.transpose(hidden_weights)
    dot = np.dot(tranposed, delta_o)
    return dot
    
def delta_output(output, error):
    # delta_o = (output * (1 - output)) * error 
    delta_o = (output * (1 - output)) * error
    return delta_o

def change_variable(delta_o, outputs):
    change = np.zeros([len(delta_o), len(outputs)])

    for i in range(len(delta_o)):
        for j in range(len(outputs)):
            change[i][j] = delta_o[i] * outputs[j]
    
    return change

def weight_adjust(weights, change, learning_rate):
    half = (learning_rate * change)
    weights = weights + half
    return weights

folds, folds_labels = create_folds(n_folds, path)
input_size = len(folds[0][0])
accuracies = []
convergence_graph = []
times = []
for i in range(n_folds):
    test = folds[i]
    test_label = folds_labels[i]
    train = np.concatenate(np.delete(folds, i, 0))
    train_label = np.concatenate(np.delete(folds_labels, i, 0))

    hidden_weights = init_weights(input_size, hidden_layer_size)
    output_weights = init_weights(hidden_layer_size, output_layer_size)
    start_time = time.time()
    for k in range(100):
        generation_error = 0
        for j in range(len(train)):
            hidden_output = activation_function(dot_product(hidden_weights, train[j]).astype(float))
            final_output = activation_function(dot_product(output_weights, hidden_output).astype(float))
            label = np.fromstring(train_label[j], sep=' ', dtype=int)
            error = network_error(label, final_output)
            generation_error += np.sum(error)
            if (final_output.argmax() != label.argmax()):
                delta_o = delta_output(final_output, error)
                change = change_variable(delta_o, hidden_output)
                output_weights = weight_adjust(output_weights, change, learning_rate)
                hidden_error = hidden_layer_error(delta_o, output_weights)
                delta_o_hidden = delta_output(hidden_output, hidden_error)
                change_hidden = change_variable(delta_o_hidden, train[j])
                hidden_weights = weight_adjust(hidden_weights, change_hidden, learning_rate)
        
        convergence_graph.append(generation_error)
        file.write(str(generation_error) + "\n")
    end_time = time.time()
    execution_time = ((end_time - start_time))
    file.write("-"*100+ "\n")
    plt.plot(convergence_graph)
    convergence_graph = []
    
    count_accuracy = 0
    confusion_matrix = np.zeros([3, 3])

    for x in range(len(test)):
        hidden_output = activation_function(dot_product(hidden_weights, test[x]).astype(float))
        final_output = activation_function(dot_product(output_weights, hidden_output).astype(float))
        label = np.fromstring(test_label[x], sep=' ', dtype=int)
        
        confusion_matrix[label.argmax()][final_output.argmax()] += 1

        if (label.argmax() == final_output.argmax()):
            count_accuracy += 1
            #print("entro argmax")
    
    print("Matriz de confusion para " + str(i + 1) + " fold")
    print(confusion_matrix)
    accuracy = np.round(count_accuracy*100/len(test), 2)
    accuracies.append(accuracy)
    print("Accuracy: " + str(accuracy))
    print("Time execution: " + str(execution_time))
    print()
    file.write(str(count_accuracy) + "\n")
    file.write("-"*100+ "\n")

plt.show()
total_accuracy = np.mean(accuracies)
print("Total accuracy: " + str(total_accuracy))








# def test_variables():
#     inputs = [1, 2, 3, 4, 1]
#     label = [0, 1, 0]

#     print("Inputs")
#     print(inputs)

#     weights1 = init_weights(len(inputs), hidden_layer_size)
#     print("hidden_layer_weigths")
#     print(weights1)

#     dot_hidden_layer = dot_product(weights1, inputs)
#     print("dot_hidden_layer")
#     print(dot_hidden_layer)

#     output_hidden = activation_function(dot_hidden_layer)
#     print("output_hidden")
#     print(output_hidden)

#     weights2 = init_weights(len(weights1), output_layer_size)
#     print("output_layer_weights")
#     print(weights2)

#     dot_output_layer = dot_product(weights2, output_hidden)
#     print("dot_output_layer")
#     print(dot_output_layer)

#     final_output = activation_function(dot_output_layer)
#     print("final_output")
#     print(final_output)

#     error = network_error(label, final_output)
#     print("network_error")
#     print(error)

#     print("delta output")
#     delta_o = delta_output(final_output, error)
#     print(delta_o)

#     print("change")
#     change = change_variable(delta_o, output_hidden)
#     #print(change)

#     print("weight adjust")
#     weights2 = weight_adjust(weights2, change, learning_rate)
#     print(weights2)

#     print("hidden error")
#     hidden_error = hidden_layer_error(delta_o, weights2)
#     print(hidden_error)

#     print("delta o hidden")
#     delta_o_hidden = delta_output(output_hidden, hidden_error)
#     print(delta_o_hidden)

#     change_hidden = change_variable(delta_o_hidden, inputs)

#     print("Weights1")
#     weights1 = weight_adjust(weights1, change_hidden, learning_rate)
#     print(weights1)

#test_variables()