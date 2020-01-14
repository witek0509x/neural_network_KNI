from random import random
from math import exp


def initialise_network(n_input, n_hidden, n_output):
    network = []
    hidden = [{'weights': [random() for _ in range(n_input + 1)]} for __ in range(n_hidden)]
    network.append(hidden)
    output = [{'weights': [random() for _ in range(n_hidden + 1)]} for __ in range(n_output)]
    network.append(output)
    return network


def activate(inputs, neuron):
    activation = neuron['weights'][-1]
    for i in range(len(inputs)):
        activation += neuron['weights'][i] * inputs[i]
    neuron['output'] = 1.0 / (1.0 + exp(-activation))
    return neuron


def forward_propagate(network, inputs):
    for i in range(len(network)):
        new_inputs = []
        for j in range(len(network[i])):
            network[i][j] = activate(inputs, network[i][j])
            new_inputs.append(network[i][j]['output'])
        inputs = new_inputs
    return inputs, network


def transfer_derivative(output):
    return output * (1 - output)


def calculate_deltas(network, expected):
    for i in reversed(range(len(network))):
        errors = []
        if len(network) - 1 == i:
            for j in range(len(network[i])):
                errors.append(expected[j] - network[i][j]['output'])
        else:
            for j in range(len(network[i])):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        for j in range(len(errors)):
            network[i][j]['delta'] = errors[j] *  transfer_derivative(network[i][j]['output'])


def update_weights(network, row, l_rate):
    for i in range(len(network)):
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        else:
            inputs = row
        for k in range(len(network[i])):
            for j in range(len(inputs)):
                network[i][k]['weights'][j] += l_rate * network[i][k]['delta'] * inputs[j]
            network[i][k]['weights'][-1] += l_rate * network[i][k]['delta']


def calculate_error(network, expected):
    error = 0.0
    for i in range(len(network[-1])):
        error += (1/2)*(network[-1][i]['output'] - expected[i])**2
    return error


num_epochs = 10000
network = initialise_network(2, 3, 2)
learning_set = [[[1, 1], [1, 0]], [[1, 0], [0, 1]], [[0, 1], [0, 1]], [[0, 0], [0, 1]]]
for epoch in range(num_epochs):
    all_error = 0.0
    for example in learning_set:
        forward_propagate(network, example[0])
        calculate_deltas(network, example[1])
        update_weights(network, example[0], 0.1)
        all_error += calculate_error(network, example[1])
    if 0 == epoch % 1000:
        print("error in epoch: " + str(epoch) + " is: " + str(all_error))
for example in learning_set:
    forward_propagate(network, example[0])
    print(example)
    for i in range(len(network[-1])):
        print("Neuron ", i, " output: ", network[-1][i]['output'])