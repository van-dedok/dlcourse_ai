import numpy as np


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''
    # TODO: Copy from previous assignment
    #raise Exception("Not implemented!")

    return  reg_strength * np.sum (W * W), 2 * reg_strength * W


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    # TODO copy from the previous assignment
    probs = softmax(predictions)
    batch_size = probs.shape[0]
    #print(probs.shape, target_index.reshape((batch_size,-1)).shape)
    
    if len(probs.shape) == 1:
        loss = cross_entropy_loss(probs,target_index)
        classes_num = probs.shape[0]
        delta_array = (target_index == np.arange(classes_num))
        dprediction = - (delta_array - probs)
        
    elif len(probs.shape) == 2:
        batch_size = probs.shape[0]
        #note the mess with shape of target_index
        loss = cross_entropy_loss(probs,target_index.reshape(batch_size,1))
        classes_num = probs.shape[1]
        delta_array = (target_index.reshape((batch_size,-1)) == np.arange(classes_num))
        dprediction = - (delta_array - probs) / batch_size

    return loss, dprediction


class Param:
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)

        
class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        # TODO copy from the previous assignment
        raise Exception("Not implemented!")

    def backward(self, d_out):
        # TODO copy from the previous assignment
        raise Exception("Not implemented!")
        return d_result

    def params(self):
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.1 * np.random.randn(n_input, n_output))
        self.B = Param(0.1 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO: Implement forward pass
        # Your final implementation shouldn't have any loops
        self.X = X.copy()
        # B here is being broadcasted to (batch_size, n_output) shape
        B = self.B.value
        W = self.W.value
        X = self.X
        return np.dot(X, W) + B

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment
        B = self.B.value
        W = self.W.value
        X = self.X
        
        batch_size = X.shape[0]
        n_output = B.shape[1]
        self.W.grad = np.dot(np.transpose(X), d_out)
        self.B.grad = np.sum(d_out,axis = 0).reshape(1,n_output)
    
        return np.dot(d_out, np.transpose(W))

    def params(self):
        return {'W': self.W, 'B': self.B}


    
class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))

        self.padding = padding
        self.X = None


    def forward(self, X):
        self.X = X.copy()
        batch_size, height, width, input_channels = X.shape
        
        padding = self.padding
        filter_size = self.filter_size
        out_channels = self.out_channels
        W = self.W.value
        B = self.B.value
        stride = 1
        
        # TODO: Implement forward pass
        # Hint: setup variables that hold the result
        # and one x/y location at a time in the loop below
        
        # It's ok to use loops for going over width and height
        # but try to avoid having any other loops
        out_height = int((height - filter_size + 2 * padding) / stride + 1.)
        out_width = int((width - filter_size + 2 * padding) / stride + 1.)
        Y = np.zeros((batch_size, out_height, out_width, out_channels))

        #W = np.ones((filter_size, filter_size, input_channels, out_channels))
        W_reshaped = W.reshape(filter_size * filter_size * input_channels, out_channels)

        X_pad = self.get_X_pad(X)
        B_broadcasted = np.concatenate([B.reshape(1,-1)] * batch_size)
        

        for y in range(out_height):
            for x in range(out_width):
                window = X_pad[:,y:y + filter_size, x:x + filter_size,:]
                window = window.reshape(batch_size, filter_size * filter_size * input_channels)
                Y[:,y,x,:] = np.dot(window, W_reshaped) + B_broadcasted
                pass
        return Y

    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients
        B = self.B.value
        W = self.W.value
        X = self.X
        filter_size = self.filter_size
        out_channels = self.out_channels
        
        batch_size, height, width, input_channels = X.shape
        _, out_height, out_width, out_channels = d_out.shape
        print(d_out.shape)

        # TODO: Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output

        # Try to avoid having any other loops here too
        

        X_pad = self.get_X_pad(X)
        W_reshaped = W.reshape(filter_size * filter_size * input_channels, out_channels)
        #self.W.grad inicialized as zeros, check if you don't trust!
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement backward pass for specific location
                # Aggregate gradients for both the input and
                # the parameters (W and B)
                window = X_pad[:,y:y + filter_size, x:x + filter_size,:]
                window = window.reshape(batch_size, filter_size * filter_size * input_channels)
                self.W.grad += np.dot(np.transpose(window), d_out[:,y,x,:]).reshape(batch_size,filter_size,filter_size,input_channels)
                pass
        print("Window shape: ", window)
        print("d_out[:,x,y,:] shape: ", d_out[:,y,x,:].shape)
        print("self.W.grad shape: ", self.W.grad.shape )
        print("")
    
        #return np.dot(d_out, np.transpose(W))
        raise Exception("Not implemented!")
        
    def get_X_pad(self, X):
        padding = self.padding
        batch_size, height, width, input_channels = X.shape
        h_zeros = np.zeros((batch_size, padding, X.shape[2], input_channels))
        X_pad = np.concatenate((h_zeros, X, h_zeros) ,axis = 1)
        v_zeros = np.zeros((batch_size, X_pad.shape[1], padding, input_channels))
        X_pad = np.concatenate((v_zeros, X_pad, v_zeros), axis = 2)
        return X_pad
    
    def params(self):
        return { 'W': self.W, 'B': self.B }


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        # TODO: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension
        raise Exception("Not implemented!")

    def backward(self, d_out):
        # TODO: Implement maxpool backward pass
        batch_size, height, width, channels = self.X.shape
        raise Exception("Not implemented!")

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape

        # TODO: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]
        raise Exception("Not implemented!")

    def backward(self, d_out):
        # TODO: Implement backward pass
        raise Exception("Not implemented!")

    def params(self):
        # No params!
        return {}
