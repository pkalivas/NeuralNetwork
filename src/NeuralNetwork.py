import numpy as np


class NeuralNetwork:


    '''
    NeuralNetwork Object:
        num_input: int <- number of features as an input value
        hidden: list(int()) <- list of the sizes of the hidden layers 
        num_output: int <- number of outputs 
        learning_rate: float <- the learning rate of gradient decent
        weights: list(np.ndarray()) <- these are the hidden layers of the network
        biases: list(np.ndarray()) <- hidden biases in the network
    '''

    def __init__(self, num_input, hidden_sizes, num_output, learning_rate = .1):
        self.num_input = num_input
        self.num_output = num_output
        self.learning_rate = learning_rate
        self.weights, self.`bias = [], []
        
        # init the network
        prev_size = num_input
        for layer in hidden_sizes:
            self.weights.append(np.random.rand(layer, prev_size))
            self.bias.append(np.random.rand(layer, 1))
            prev_size = layer
        self.weights.append(np.random.rand(num_output, hidden_sizes[::-1][0]))
        self.bias.append(np.random.rand(num_output, 1))

        # activation / deactivation functions
        self.act = {'sigmoid' : lambda x: 1 / (1 + np.exp(-x)), 
                    'dsigmoid' : lambda x: x * (1 - x), 
                    'relu' : lambda x: max(0, x),
                    'drelu' : lambda x: 1 if x > 0 else 0}



    '''overridden functions'''
    def __repr__(self):
        '''represent a few stats of the network'''
        return 'in={0}  w={1}  o={2}'.format(self.num_input, ','.join([str(len(w)) for w in self.weights]), self.num_output)



    '''ai functions'''
    def predict(self, inputs):
        '''predict a single input'''
        __, prediction = self.feed_forward(inputs)
        return prediction



    def train(self, train, target, iterations = 100):
        '''iter over train and forward propagate through the network, then backpropagate the error'''
        for i in range(iterations):
            for index, row in enumerate(train):
                self.backpropagate(row, target[index])



    def backpropagate(self, inputs, target):
        '''adjust the weights and biases in the network by backpropagating the error gradient'''
        hidden_outputs, prediction = self.feed_forward(inputs)
        error = target - prediction
        for i in range(len(hidden_outputs) - 1, -1, -1):
            prev_out = hidden_outputs[i - 1] if i > 0 else inputs
            hidden_error = np.dot(np.transpose(self.weights[i]), error)
            error = np.multiply(error, self.learning_rate)
            gradient = np.vectorize(self.act['dsigmoid'])(hidden_outputs[i])
            step = np.multiply(error, gradient)
            weight_delta = np.dot(step, np.transpose(prev_out))

            self.weights[i] += weight_delta
            self.bias[i] += step
            error = hidden_error



    def feed_forward(self, inputs):
        '''forward propagate an input through the layers of the network'''
        hidden_outputs = []
        for i, layer in enumerate(self.weights):
            layer_output = np.add(np.dot(layer, inputs), self.bias[i])
            for j in range(len(layer_output)):
                for k in range(len(layer_output[j])):
                    layer_output[j][k] = self.act['sigmoid'](layer_output[j][k])
            hidden_outputs.append(layer_output)
            inputs = layer_output
        return hidden_outputs, layer_output
