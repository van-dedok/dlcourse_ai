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
    loss = cross_entropy_loss(probs,target_index)
    if len(probs.shape) == 1:
        classes_num = probs.shape[0]
        delta_array = (target_index == np.arange(classes_num))
        dprediction = - (delta_array - probs)
    elif len(probs.shape) == 2:
        batch_size = probs.shape[0]
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
    

def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''
    batch_size = target_index.shape[0]
    target_index = target_index.reshape(batch_size,-1)
    predictions = np.dot(X, W)
    loss, dpredictions = softmax_with_cross_entropy(predictions, target_index)
    
    dW = np.dot(np.transpose(X),dpredictions)
    # TODO implement prediction and gradient over W
    # Your final implementation shouldn't have any loops
    #raise Exception("Not implemented!")
    
    return loss, dW


class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1, verbose = 1):
        '''
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''
        loss_history = []
        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            num_sections = sections.shape[0]
            batches_indices = np.array_split(shuffled_indices, sections)
            #X_batch = np.take_along_axis(X, batches_indices,reshape(, axis = 0)
            for section in range(num_sections):
                X_batch = np.take_along_axis(X, batches_indices[section].reshape(batch_size, -1), axis = 0 )
                y_batch = np.take_along_axis(y, batches_indices[section], axis = 0 )
                loss, dW = linear_softmax(X_batch, self.W, y_batch)
                loss_reg, dW_reg = l2_regularization(self.W, reg)
                loss += loss_reg
                dW += dW_reg
                self.W = self.W - learning_rate * dW
            # TODO implement generating batches from indices
            # Compute loss and gradients
            # Apply gradient to weights using learning rate
            # Don't forget to add both cross-entropy loss
            # and regularization!
            #raise Exception("Not implemented!")

            # end
            if verbose == 1:
                print("Epoch %i, loss: %f, l2_loss %f" % (epoch, loss, loss_reg))
            loss_history.append(loss)

        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''
        #y_pred = np.zeros(X.shape[0], dtype=np.int)
        probs = np.dot(X, self.W)
        y_pred = np.argmax(probs, axis = 1)
        print(y_pred.shape)
        # TODO Implement class prediction
        # Your final implementation shouldn't have any loops
       

        return y_pred



                
                                                          

            

                
