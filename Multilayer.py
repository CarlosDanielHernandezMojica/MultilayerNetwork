import numpy as np
import pandas as pd
from sklearn.metrics import multilabel_confusion_matrix

## Ideas 
## Normalize data

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
    #print(df_shuffled)

    inputs = df_shuffled.iloc[:, : -1]
    labels = df_shuffled.iloc[:, -1]

    max_values = inputs[:].max()
    min_values = inputs[:].min()
    
    # print(max_values)
    # print(min_values)

    inputs = (inputs - min_values) / (max_values - min_values)
    # inputs["bias"] = 1
    # print(inputs)
    

    inputs = np.array(inputs)
    labels = np.array(labels)

    # print(inputs)
    # print(len(inputs))
    # print(labels)
    # print(len(labels))

    folds = np.split(inputs, n_folds)
    label_folds = np.split(labels, n_folds)

    # print(folds)
    # print(len(folds))
    # print(label_folds)
    # print(len(label_folds))
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
    # outputs = np.empty(len(dot_products))
    
    # for i in range(len(dot_products)):
    #     outputs[i] = 1/(1 + np.exp(-(dot_products[i])))
    outputs = 1 / (1 + np.exp(-dot_products))
    return outputs

def network_error(label, output):

    error = (np.mean(np.square(label - output)))
    #error = (output - label)**2
    return error

def hidden_layer_error(delta_o, hidden_weights):
    tranposed = np.transpose(hidden_weights)
    dot = np.dot(tranposed, delta_o)
    return dot
    
def delta_output(output, error):
    delta_o = (output * (1 - (output))) * error 
    # print("Delta o")
    # print(delta_o)
    
    return delta_o

def change_variable(delta_o, outputs):
    change = np.zeros([len(delta_o), len(outputs)])
    # print("empty change")
    # print(change)

    for i in range(len(delta_o)):
        for j in range(len(outputs)):
            change[i][j] = delta_o[i] * outputs[j]
            #change[i][j] = delta_o[i] * outputs[i][j]


    # print("change")
    # print(np.round(change, 3))
    
    return change

def weight_adjust(weights, change, learning_rate):
    
    half = (learning_rate * change)
    # print("learnig_rate * change")
    # print(half)
    weights = weights + half
    # print("weights adjusted")
    # print(weights)
    return weights

def training(folds, label_folds, hidden_layer_size, output_layer_size, lr):

    n_inputs = len(folds[0][0])
    
    for i in range(5):
        test_fold = folds[i]
        test_label = label_folds[i]
        training_folds = np.concatenate(np.delete(folds, i, 0))
        training_labels = np.concatenate(np.delete(label_folds, i, 0))
        

        hidden_weights = init_weights(n_inputs, hidden_layer_size)
        output_weights = init_weights(hidden_layer_size, output_layer_size)
        
        for i in range(50):
            total_error = 0
            for j in range(len(training_folds)):
                dot_hidden = dot_product(hidden_weights, training_folds[j])
                output_hidden = activation_function(dot_hidden.astype(float))
                dot_output = dot_product(output_weights, output_hidden)
                final_output = activation_function(dot_output.astype(float))
                # print(final_output)
                idxmax = final_output.argmax()
                final_label = np.zeros_like(final_output)
                final_label[idxmax] = 1
                # print(final_label)
                label = np.fromstring(training_labels[j], sep=' ', dtype=int)
                # print(label)
                # Check this -----------
                error = network_error(label, final_output)
                # print(error)
                # print(error)
                sum_error = np.sum(error)
                print(sum_error)
                total_error += sum_error
                #file.write(str(sum_error) + "\n")
                # ----------------------
                if not (np.array_equal(final_label, label)):
                    print("Training...")
                    delta_o = delta_output(final_output, error)
                    #change = change_variable(delta_o, output_hidden)
                    change = change_variable(delta_o, output_hidden)
                    output_weights = weight_adjust(output_weights, change, lr)
                    hidden_error = hidden_layer_error(delta_o, output_weights)
                    delta_o_hidden = delta_output(output_hidden, hidden_error)
                    change_hidden = change_variable(delta_o_hidden, training_folds[j])
                    hidden_weights = weight_adjust(hidden_weights, change_hidden, lr)

            file.write(str(total_error)+ "\n")
            print(total_error)
        print("-"*100 + "\n")
        
        predicted_labels = []
        original_labels = []
        for k in range(len(test_fold)):
            dot_hidden = dot_product(hidden_weights, test_fold[k])
            output_hidden = activation_function(dot_hidden.astype(float))
            dot_output = dot_product(output_weights, output_hidden)
            final_output = activation_function(dot_output.astype(float))
            idxmax = final_output.argmax()
            final_label = np.zeros_like(final_output)
            final_label[idxmax] = 1

            predicted_labels.append(final_label)
            
            label = np.fromstring(test_label[k], sep=' ')
            
            original_labels.append(label)

        # This could be useful
        #print(predicted_labels)
        #print(original_labels)
        # cm = multilabel_confusion_matrix(original_labels.argmax(axis=1), predicted_labels.argmax(axis=1))
        #print(cm)
        #print(counter)
        #file.write(str(counter))
        file.write("-"*100 + "\n")
    
    
    return 0

folds, label_folds = create_folds(5, path)
# print(folds)
# print(label_folds)
training(folds, label_folds, hidden_layer_size, output_layer_size, learning_rate)


def test_variables():
    inputs = [1, 2, 3, 4, 1]
    label = [0, 1, 0]

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

    print("delta output")
    delta_o = delta_output(final_output, error)
    print(delta_o)

    print("change")
    change = change_variable(delta_o, output_hidden)
    #print(change)

    print("weight adjust")
    weights2 = weight_adjust(weights2, change, learning_rate)
    print(weights2)

    print("hidden error")
    hidden_error = hidden_layer_error(delta_o, weights2)
    print(hidden_error)

    print("delta o hidden")
    delta_o_hidden = delta_output(output_hidden, hidden_error)
    print(delta_o_hidden)

    change_hidden = change_variable(delta_o_hidden, inputs)

    print("Weights1")
    weights1 = weight_adjust(weights1, change_hidden, learning_rate)
    print(weights1)

# test_variables()

## Ideas to do
## Make a before dot method
## The change variable uses the value of each line connected to the next neuron
## 










