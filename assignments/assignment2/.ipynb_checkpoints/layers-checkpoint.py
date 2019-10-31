import numpy as np


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    # TODO implement softmax
    # Your final implementation shouldn't have any loops

    preds = predictions.copy()
    
    if len(preds.shape) == 1:
        preds -= np.max(preds)
        exp_array = np.exp(preds)
        return exp_array / np.sum(exp_array)
    elif len(preds.shape) == 2:
        batch_size = preds.shape[0]
        preds -= np.max(preds,axis = 1).reshape(batch_size, -1)
        exp_array = np.exp(preds)
        return exp_array / np.sum(exp_array, axis = 1).reshape((batch_size,-1))
    raise Exception("Wrong shape of predictions array!")
    
    

def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    # TODO implement cross-entropy
    # Your final implementation shouldn't have any loops
    
    if len(probs.shape) == 1:
        return -np.log(probs[target_index])
    elif len(probs.shape) == 2:
            batch_size = probs.shape[0]
            relevant_probs = np.take_along_axis(probs, target_index, axis = 1)
            relevant_probs = np.where(relevant_probs == 0., 1e-100, relevant_probs)
            return -np.sum(np.log(relevant_probs)) / batch_size
    raise Exception("Wrong shape of probs array!!")


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
    # TODO implement softmax with cross-entropy
    # Your final implementation shouldn't have any loops
  
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

    # TODO: implement l2 regularization and gradient
    # Your final implementation shouldn't have any loops
    
    return  reg_strength * np.sum (W * W), 2 * reg_strength * W


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)
        
class ReLULayer:
    def __init__(self):
        self.X_saved = None
        self.der_at_zero = 0.5
        pass

    def forward(self, X):
        # TODO: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        self.X_saved = (X == np.zeros_like(X)) * self.der_at_zero + (X > np.zeros_like(X))
        
        return X * self.X_saved

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Your final implementation shouldn't have any loops
        d_result = d_out * self.X_saved
        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
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
