import numpy as np

class NeuralNetwork(object):
    def __init__(self, D, k, layers, step_size = 1e-0, reg = 1e-3):
        """

        :param D: dimension of input data
        :param k: # of classes to predict
        :param layers: list of number of units in each hidden layer (length specifies # of layers)
        """
        self.D = D
        self.k = k
        self.layers = {}
        self.step_size = step_size
        self.reg = reg

        input_dim = D
        for l, layer in enumerate(layers):
            #  Initialize each of the hidden layers
            self.layers[l] = {'W': 0.01 * np.random.randn(input_dim, layer),
                              'b': np.zeros((1, layer))}
            input_dim = layer

        self.layers[len(layers)] = {'W': 0.01 * np.random.randn(input_dim, k),
                              'b': np.zeros((1, k))}


    def step(self, X_train, y_train):
        num_examples = X_train.shape[0]
        hidden_out = {}
        out = X_train
        for l in range(len(self.layers)-1):
            layer = self.layers[l]
            out = np.maximum(0, np.dot(out, layer['W']) + layer['b'])
            hidden_out[l] = out

        out_layer = self.layers[len(self.layers)-1]
        scores = np.dot(out, out_layer['W'] + out_layer['b'])

        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        correct_logprobs = -np.log(probs[range(num_examples), y_train])
        data_loss = np.sum(correct_logprobs) / num_examples
        reg_loss = sum([0.5 * self.reg * np.sum(l['W'] * l['W']) for l in self.layers.values()])
        loss = data_loss + reg_loss


        dscores = probs
        dscores[range(num_examples), y_train] -= 1
        dscores /= num_examples

        dw = {}
        dhidden = {}

        dl = dscores

        for l in range(1, len(self.layers))[::-1]:
            backpass = np.dot(hidden_out[l-1].T, dl)
            dw[l] = {'W':  backpass,
                     'b': np.sum(dl, axis=0, keepdims=True)}
            dhidden[l] = np.dot(dl, self.layers[l]['W'].T)
            dhidden[l][hidden_out[l-1] <= 0] = 0

            dl = dhidden[l]

        dw[0] = {'W': np.dot(X_train.T, dl),
               'b': np.sum(dl, axis=0, keepdims=True)}

        for l in range(len(self.layers)):
            dw[l]['W'] += self.reg * self.layers[l]['W']

            self.layers[l]['W'] += -self.step_size*dw[l]['W']
            self.layers[l]['b'] += -self.step_size*dw[l]['b']

        return loss


    def train(self, X_train, y_train):
        """

        :param X_train:
        :param y_train:
        :return:
        """
        # loop through layers w/ forward pass, backward pass
        pass

    def predict(self, X_test):
        output = X_test
        for l in range(len(self.layers) - 1):
            layer = self.layers[l]
            output = np.maximum(0, np.dot(output, layer['W']) + layer['b'])

        out_layer = self.layers[l + 1]
        scores = np.dot(output, out_layer['W'] + out_layer['b'])
        predicted_class = np.argmax(scores, axis=1)

        return predicted_class

    def get_accuracy(self, X_test, y_test):
        predicted_class = self.predict(X_test)

        return (np.mean(predicted_class == y_test))