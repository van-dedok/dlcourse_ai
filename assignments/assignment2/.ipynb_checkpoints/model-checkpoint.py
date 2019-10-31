import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        self.layers = dict()
        self.layers['hidden'] = FullyConnectedLayer(n_input, hidden_layer_size)
        self.layers['hidden_ReLU'] = ReLULayer()
        self.layers['output'] = FullyConnectedLayer( hidden_layer_size, n_output)
        self.layers['output_ReLU'] = ReLULayer()
        
        self.layers_order = ['hidden',
                        'hidden_ReLU',
                        'output',
                        'output_ReLU'
                       ]
        #raise Exception("Not implemented!")

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
        
        self.nullify_grads()
        
        #raise Exception("Not implemented!")
        
        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
        layers = self.layers
        params = self.params()
        '''
        temp_matrix = layers['hidden'].forward(X)
        temp_matrix = layers['hidden_ReLU'].forward(X)
        temp_matrix = layers['output'].forward(temp_matrix)
        temp_matrix = layers['output_ReLU'].forward()
        '''
        
        temp_matrix = X
        for layer_name in self.layers_order:
            temp_matrix = layers[layer_name].forward(temp_matrix)
        batch_size = y.shape[0]
        batch_size = y.shape[0]
        loss, dpreds = softmax_with_cross_entropy(temp_matrix, y)
        
        temp_matrix = dpreds
        for layer_name in self.layers_order[::-1]:
            temp_matrix = layers[layer_name].backward(temp_matrix)
        
        # add regularization directly to each parameter
        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        raise Exception("Not implemented!")

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        pred = np.zeros(X.shape[0], np.int)

        raise Exception("Not implemented!")
        return pred

    def params(self):
        result = {}
        layers = self.layers
        #here the "result" variable is dict of dicts -- first key refers
        # to layer, second -- to the parametr
        result = {key: layers[key].params() for key in layers.keys()}
        # TODO Implement aggregating all of the params
        return result
    
    def nullify_grads(self):
        layers = self.layers
        params = self.params()
        for layer in layers.keys():
            for param in params[layer].keys():
                #grad_to_nullify = params[layer][param].grad
                params[layer][param].grad = np.zeros_like(params[layer][param].grad)
        return 0
