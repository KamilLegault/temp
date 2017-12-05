# coding: utf8
from sklearn import datasets, model_selection, metrics 
from matplotlib.colors import colorConverter, ListedColormap
import matplotlib.pyplot as plt
import itertools
import collections
import numpy as np 
import pickle, gzip
import pylab
import time
from pylab import rcParams
rcParams['figure.figsize'] = 12, 8

#####################################################################################################################################################
#####################################################################################################################################################
#This work is based on the neural network implementation by Peter Roeleants##########################################################################
#https://github.com/peterroelants/peterroelants.github.io/blob/master/notebooks/neural_net_implementation/neural_network_implementation_part05.ipynb#
#some code adapted from https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/ and course demo##################################
#####################################################################################################################################################
#####################################################################################################################################################

def gridplot(layers,train,test,plotIndex,n_points=50,title='no title'):
    ax = pylab.subplot(plotIndex)
    ax.set_title(title)
    train_test = np.vstack((train,test))
    (min_x1,max_x1) = (min(train_test[:,0]),max(train_test[:,0]))
    (min_x2,max_x2) = (min(train_test[:,1]),max(train_test[:,1]))

    xgrid = np.linspace(min_x1,max_x1,num=n_points)
    ygrid = np.linspace(min_x2,max_x2,num=n_points)
    thegrid = np.array(combine(xgrid,ygrid))

    les_comptes = fprop(thegrid, layers) #classifieur.compute_predictions(thegrid)
    classesPred = np.argmax(les_comptes[-1],axis=1)
    
    # La grille
    pylab.scatter(thegrid[:,0],thegrid[:,1],c = classesPred, edgecolors = 'black', s=n_points)
    pylab.scatter(train[:,0], train[:,1], c = train[:,-1], marker = 'v', edgecolors = 'black', s=(n_points*1))
    pylab.scatter(test[:,0], test[:,1], c = test[:,-1], marker = 's', edgecolors = 'black', s=(n_points*1))

    h1 = pylab.plot([min_x1], [min_x2], marker='o', c = 'w',ms=2) 
    h2 = pylab.plot([min_x1], [min_x2], marker='v', c = 'w',ms=2) 
    h3 = pylab.plot([min_x1], [min_x2], marker='s', c = 'w',ms=2) 
    handles = [h1,h2,h3]

    labels = ['grille','train','test']
    pylab.legend(handles,labels)

    pylab.axis('equal')
    pylab.savefig('plots.png')
#     pylab.show()
    return pylab


def multipleGridPlot(classifierLayers, X_train, T_train, X_test, T_test, titles):
    plotIndex=221; i=0
    for i, classifier in enumerate(classifierLayers):
        if(plotIndex>=224): 
            break
        gridplot(classifier, np.c_[X_train, T_train], np.c_[X_test, T_test],plotIndex, n_points=50, title=titles[i])
        plotIndex+=1; i+=1
    gridplot(classifierLayers[i-1], np.c_[X_train, T_train], np.c_[X_test, T_test],plotIndex, n_points=50, title=titles[i-1]).show()


## http://code.activestate.com/recipes/302478/
def combine(*seqin):
    '''returns a list of all combinations of argument sequences.
for example: combine((1,2),(3,4)) returns
[[1, 3], [1, 4], [2, 3], [2, 4]]'''
    def rloop(seqin,listout,comb):
        '''recursive looping function'''
        if seqin:                       # any more sequences to process?
            for item in seqin[0]:
                newcomb=comb+[item]     # add next item to current comb
                # call rloop w/ rem seqs, newcomb
                rloop(seqin[1:],listout,newcomb)
        else:                           # processing last sequence
            listout.append(comb)        # comb finished, add to list
    listout=[]                      # listout initialization
    rloop(seqin,listout,[])         # start recursive process
    return listout


def plot_function(train_data, weights, bias, title):
    plt.figure()
    d1 = train_data[train_data[:, -1] > 0]
    d2 = train_data[train_data[:, -1] < 0]
    plt.scatter(d1[:, 0], d1[:, 1], c='b', label='classe +1')
    plt.scatter(d2[:, 0], d2[:, 1], c='g', label='classe -1')
    x = np.linspace(-10, 10, 100)
    y = -(weights[0]*x + bias)/weights[1]
    plt.plot(x, y, c='r', lw=2, label='y = -(w1*x + b1)/w2')
    plt.xlim(np.min(train_data[:, 0]) - 0.5, np.max(train_data[:, 0]) + 0.5)
    plt.ylim(np.min(train_data[:, 1]) - 0.5, np.max(train_data[:, 1]) + 0.5)
    plt.grid()
    plt.legend(loc='lower right')
    plt.title(title)
    plt.show()



def plot_error(nb_of_batches, nb_of_iterations, training_errors, validation_errors, test_errors):
    minibatch_x_inds = np.linspace(0, nb_of_iterations, num=nb_of_iterations * nb_of_batches)
    iteration_x_inds = np.linspace(1, nb_of_iterations, num=nb_of_iterations) # Plot the cost over the iterations
    plt.plot(iteration_x_inds, training_errors, 'r-', linewidth=2, label='error full training set')
    plt.plot(iteration_x_inds, validation_errors, 'b-', linewidth=3, label='error validation set')
    plt.plot(iteration_x_inds, test_errors, 'k-', linewidth=0.5, label='error test set') # Add labels to the plot
    plt.xlabel('iteration')
    plt.ylabel('$\\xi$', fontsize=15)
    plt.title('Decrease of error over backprop iteration')
    plt.legend()
    x1, x2, y1, y2 = plt.axis()
    plt.axis((0, nb_of_iterations, 0, 0.5))
    plt.grid()
    plt.show()

def plot_cost(nb_of_batches, nb_of_iterations, minibatch_costs, training_costs, validation_costs):
    # Plot the minibatch, full training set, and validation costs
    minibatch_x_inds = np.linspace(0, nb_of_iterations, num=nb_of_iterations * nb_of_batches)
    iteration_x_inds = np.linspace(1, nb_of_iterations, num=nb_of_iterations) # Plot the cost over the iterations
    plt.plot(minibatch_x_inds, minibatch_costs, 'k-', linewidth=0.5, label='cost minibatches')
    plt.plot(iteration_x_inds, training_costs, 'r-', linewidth=2, label='cost full training set')
    plt.plot(iteration_x_inds, validation_costs, 'b-', linewidth=3, label='cost validation set') # Add labels to the plot
    plt.xlabel('iteration')
    plt.ylabel('$\\xi$', fontsize=15)
    plt.title('Decrease of cost over backprop iteration')
    plt.legend()
    x1, x2, y1, y2 = plt.axis()
    plt.axis((0, nb_of_iterations, 0, 2.5))
    plt.grid()
    plt.show()

def plot_confusion_table(T_test, X_test, layers):
    y_true = np.argmax(T_test, axis=1)  # Get the target outputs
    activations = fprop(X_test, layers)  # Get activation of test samples
    y_pred = np.argmax(activations[-1], axis=1)  # Get the predictions made by the network
    conf_matrix = metrics.confusion_matrix(y_true, y_pred, labels=None)  # Get confustion matrix
    # Plot the confusion table
    class_names = ['${:d}$'.format(x) for x in range(0, 10)]  # Digit class names
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # Show class labels on each axis
    ax.xaxis.tick_top()
    major_ticks = range(0,10)
    minor_ticks = [x + 0.5 for x in range(0, 10)]
    ax.xaxis.set_ticks(major_ticks, minor=False)
    ax.yaxis.set_ticks(major_ticks, minor=False)
    ax.xaxis.set_ticks(minor_ticks, minor=True)
    ax.yaxis.set_ticks(minor_ticks, minor=True)
    ax.xaxis.set_ticklabels(class_names, minor=False, fontsize=15)
    ax.yaxis.set_ticklabels(class_names, minor=False, fontsize=15)
    # Set plot labels
    ax.yaxis.set_label_position("right")
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    fig.suptitle('Confusion table', y=1.03, fontsize=15)
    # Show a grid to seperate digits
    ax.grid(b=True, which=u'minor')
    # Color each grid cell according to the number classes predicted
    ax.imshow(conf_matrix, interpolation='nearest', cmap='binary')
    # Show the number of samples in each cell
    for x in range(conf_matrix.shape[0]):
        for y in range(conf_matrix.shape[1]):
            color = 'w' if x == y else 'k'
            ax.text(x, y, conf_matrix[y,x], ha="center", va="center", color=color)       
    plt.show()

#############################################################################################################################################
#############################################################################################################################################
#############################################################################################################################################

def load_data_with_pickle(picklefile,nbOfTraingSampleToLoad=-1):
    np.random.seed(0)
    # digits = datasets.load_digits()
    try:
        with gzip.open(picklefile,'rb') as f:
            train_set0, valid_set0, test_set0=pickle.load(f, encoding='latin1')
    except Exception:
        with gzip.open(picklefile,'rb') as f:
            train_set0, valid_set0, test_set0=pickle.load(f)
    #     digits = pickle.load(f, encoding='latin1')
    # T = np.zeros((digits.target.shape[0],10))
    n_train=np.asarray(train_set0[1]).shape[0]
    if nbOfTraingSampleToLoad > 0 and nbOfTraingSampleToLoad < n_train:
        n_train=nbOfTraingSampleToLoad
    n_skipped_col=0
    mnist_data=np.asarray(train_set0[0])
    # mnist_target=np.asarray(train_set0[1])
    
    #convert targets to onehot
    mnist_target = np.zeros((np.asarray(train_set0[1][:n_train]).shape[0],10))
    mnist_target[np.arange(len(mnist_target)), np.asarray(train_set0[1][:n_train])] += 1
    indices = np.random.permutation(len(mnist_target))
    X_train = mnist_data[indices]
    T_train = mnist_target[indices]
     
    if n_train>np.asarray(valid_set0[1]).shape[0]:
        n_train=np.asarray(valid_set0[1]).shape[0]
    T_validation = np.zeros((np.asarray(valid_set0[1][:n_train]).shape[0],10))
    T_validation[np.arange(len(T_validation)), np.asarray(valid_set0[1][:n_train])] += 1
    indices = np.random.permutation(len(T_validation))
    X_validation = np.asarray(valid_set0[0])[indices]
    T_validation = T_validation[indices]
    if n_train>np.asarray(test_set0[1]).shape[0]:
        n_train=np.asarray(test_set0[1]).shape[0]
    T_test = np.zeros((np.asarray(test_set0[1][:n_train]).shape[0],10))
    T_test[np.arange(len(T_test)), np.asarray(test_set0[1][:n_train])] += 1
    indices = np.random.permutation(len(T_test))
    T_test = T_test[indices]
    X_test = np.asarray(test_set0[0])[indices]
     
    return X_train, X_validation, X_test, T_train, T_validation, T_test


def load_data_from_file(textfile):
    moon_data = np.loadtxt(textfile)
    mnist_data = moon_data[:,:-1]
    
    n_train=np.asarray(moon_data[:,-1]).shape[0]
    n_valid_start=int(n_train*0.6)
    n_test_start=n_valid_start+int(n_train*0.2)
    n_test_end=n_test_start+int(n_train*0.2)
    
    mnist_target = np.zeros((np.asarray(moon_data[:,-1][:n_train]).shape[0],2))
    mnist_target[np.arange(len(mnist_target)), np.asarray(moon_data[:,-1][:n_train]).astype(int)] += 1
    indices = np.random.permutation(len(mnist_target))
    
    X_train = mnist_data[indices][:n_valid_start]
    X_validation = mnist_data[indices][n_valid_start:n_test_start]
    X_test = mnist_data[indices][n_test_start:n_test_end]
    T_train = mnist_target[indices][:n_valid_start]
    T_validation = mnist_target[indices][n_valid_start:n_test_start]
    T_test = mnist_target[indices][n_test_start:n_test_end]
    return X_train, X_validation, X_test, T_train, T_validation, T_test

#############################################################################################################################################
#############################################################################################################################################
# Define the non-linear functions used
def logistic(z): 
    return 1 / (1 + np.exp(-z))

def logistic_deriv(y): 
    return np.multiply(y, (1 - y))
    
def softmax(z): 
    return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

def stablesoftmax(z):
    #https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
    """Compute the softmax of vector x in a numerically stable way."""
    shiftx = z - np.max(z)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)

def naiveSoftmax(z): 
    jacob=np.diag(z)
    for i in range(len(jacob)):
        for j in range (len(jacob)):
            if i==j: jacob[i][j]=z[i]*(1-z[i])
            else: jacob[i][j]=z[i]*z[j]
    return jacob

def relu(x):
    return np.maximum(x,0)
#     return np.maximum(x,1e-16)

def relu_deriv(x):
    return np.where(x>0,1,0)

#############################################################################################################################################
#############################################################################################################################################
# Define the layers used in this model
class Layer(object):
    """Base class for the different layers.
    Defines base methods and documentation of methods."""
    
    def get_params_iter(self):
        """Return an iterator over the parameters (if any).
        The iterator has the same order as get_params_grad.
        The elements returned by the iterator are editable in-place."""
        return []
    
    def get_params_grad(self, X, output_grad):
        """Return a list of gradients over the parameters.
        The list has the same order as the get_params_iter iterator.
        X is the input.
        output_grad is the gradient at the output of this layer.
        """
        return []
    
    def get_output(self, X):
        """Perform the forward step linear transformation.
        X is the input."""
        pass
    
    def get_input_grad(self, Y, output_grad=None, T=None):
        """Return the gradient at the inputs of this layer.
        Y is the pre-computed output of this layer (not needed in this case).
        output_grad is the gradient at the output of this layer 
         (gradient at input of next layer).
        Output layer uses targets T to compute the gradient based on the 
         output error instead of output_grad"""
        pass


class LinearLayer(Layer):
    """The linear layer performs a linear transformation to its input."""
    
    def __init__(self, n_in, n_out):
        """Initialize hidden layer parameters.
        n_in is the number of input variables.
        n_out is the number of output variables."""
        self.W = np.random.randn(n_in, n_out) * 0.1
        self.b = np.zeros(n_out)
        
    def get_params_iter(self):
        """Return an iterator over the parameters."""
        return itertools.chain(np.nditer(self.W, op_flags=['readwrite']),
                               np.nditer(self.b, op_flags=['readwrite']))
    
    def get_output(self, X):
        """Perform the forward step linear transformation."""
        return X.dot(self.W) + self.b
        
    def get_params_grad(self, X, output_grad):
        #TODO add web source
        """Return a list of gradients over the parameters."""
        JW = X.T.dot(output_grad)
        Jb = np.sum(output_grad, axis=0)
#         return [g for g in itertools.chain(np.nditer(JW), np.nditer(Jb))]
        return [g for g in (itertools.chain(np.nditer(JW), np.nditer(Jb)))]
    
    def get_input_grad(self, Y, output_grad):
        """Return the gradient at the inputs of this layer."""
        return output_grad.dot(self.W.T)


class LogisticLayer(Layer):
    """The logistic layer applies the logistic function to its inputs."""
    
    def get_output(self, X):
        """Perform the forward step transformation."""
        return logistic(X)
    
    def get_input_grad(self, Y, output_grad):
        """Return the gradient at the inputs of this layer."""
        return np.multiply(logistic_deriv(Y), output_grad)


class ReluLayer(Layer):
    """The logistic layer applies the logistic function to its inputs."""
    
    def get_output(self, X):
        """Perform the forward step transformation."""
        return relu(X)
    
    def get_input_grad(self, Y, output_grad):
        """Return the gradient at the inputs of this layer."""
        return np.multiply(relu(Y), output_grad)


class NaiveSoftmaxOutputLayer(Layer):
    """The softmax output layer computes the classification propabilities at the output."""
    
    def get_output(self, X):
        """Perform the forward step transformation."""
#         return naiveSoftmax(X)
        return softmax(X)
#         return stablesoftmax(X)
    
    def get_input_grad(self, Y, T):
        """Return the gradient at the inputs of this layer."""
        return (Y - T) / Y.shape[0]
    
    def get_cost(self, Y, T):
        """Return the cost at the output of this output layer."""
        sum0=np.zeros(Y.shape)
        for t in range(T.shape[0]):
#         for t in T:
            for y in Y:
                sum0[t]+=T[t]*(np.log(y))            
        return -sum0.sum()/Y.shape[0]**2
#         return - np.multiply(T, np.log(Y)).sum() / Y.shape[0]

class VectorizedSoftmaxOutputLayer(Layer):
    """The softmax output layer computes the classification propabilities at the output."""
    
    def get_output(self, X):
        """Perform the forward step transformation."""
        return softmax(X)
#         return stablesoftmax(X)
    
    def get_input_grad(self, Y, T):
        """Return the gradient at the inputs of this layer."""
        return (Y - T) / Y.shape[0]
    
    def get_cost(self, Y, T):
        """Return the cost at the output of this output layer."""
        return - np.multiply(T, np.log(Y)).sum() / Y.shape[0]

#############################################################################################################################################
#############################################################################################################################################

# Define the forward propagation step as a method.
def fprop(input_samples, layers):
    """
    Compute and return the forward activation of each layer in layers.
    Input:
        input_samples: A matrix of input samples (each row is an input vector)
        layers: A list of Layers
    Output:
        A list of activations where the activation at each index i+1 corresponds to
        the activation of layer i in layers. activations[0] contains the input samples.  
    """
    activations = [input_samples] # List of layer activations
    # Compute the forward activations for each layer starting from the first
    X = input_samples
    for layer in layers:
        Y = layer.get_output(X)  # Get the output of the current layer
        activations.append(Y)  # Store the output for future processing
        X = activations[-1]  # Set the current input as the activations of the previous layer
    return activations  # Return the activations of each layer


# Define the backward propagation step as a method
def bprop(activations, targets, layers):
    """
    Perform the backpropagation step over all the layers and return the parameter gradients.
    Input:
        activations: A list of forward step activations where the activation at 
            each index i+1 corresponds to the activation of layer i in layers. 
            activations[0] contains the input samples. 
        targets: The output targets of the output layer.
        layers: A list of Layers corresponding that generated the outputs in activations.
    Output:
        A list of parameter gradients where the gradients at each index corresponds to
        the parameters gradients of the layer at the same index in layers. 
    """
    param_grads = collections.deque()  # List of parameter gradients for each layer
    output_grad = None  # The error gradient at the output of the current layer
    # Propagate the error backwards through all the layers.
    #  Use reversed to iterate backwards over the list of layers.
    for layer in reversed(layers):   
        Y = activations.pop()  # Get the activations of the last layer on the stack
        # Compute the error at the output layer.
        # The output layer error is calculated different then hidden layer error.
        if output_grad is None:
            input_grad = layer.get_input_grad(Y, targets)
        else:  # output_grad is not None (layer is not output layer)
            input_grad = layer.get_input_grad(Y, output_grad)
        # Get the input of this layer (activations of the previous layer)
        X = activations[-1]
        # Compute the layer parameter gradients used to update the parameters
        grads = layer.get_params_grad(X, output_grad)
        param_grads.appendleft(grads)
        # Compute gradient at output of previous layer (input of current layer):
        output_grad = input_grad
    return list(param_grads)  # Return the parameter gradients



# ### SGD updates
# Define a method to update the parameters
def update_params(layers, param_grads, learning_rate):
    """
    Function to update the parameters of the given layers with the given gradients
    by gradient descent with the given learning rate.
    """
    for layer, layer_backprop_grads in list(zip(layers, param_grads)):
        for param, grad in zip(layer.get_params_iter(), layer_backprop_grads):
            # The parameter returned by the iterator point to the memory space of
            #  the original layer and can thus be modified inplace.
            param -= learning_rate * grad  # Update each parameter

    
def get_err_rate(T_test,activations):
    y_true = np.argmax(T_test, axis=1)  # Get the target outputs
    y_pred = np.argmax(activations[-1], axis=1)  # Get the predictions made by the network
    return (1-np.mean(y_true == y_pred))


def gradient_check(layers, X_temp, T_temp, layersFrom2nClassifier=None):
    activations = fprop(X_temp, layers)
    paramGrads = bprop(activations, T_temp, layers)
    
    if layersFrom2nClassifier is not None:
        activationsOf2nClassifier = fprop(X_temp, layersFrom2nClassifier)
        paramGradsOf2nClassifier = bprop(activationsOf2nClassifier, T_temp, layersFrom2nClassifier)
        
# Set the small change to compute the numerical gradient
    eps = 1e-4
# Compute the numerical gradients of the parameters in all layers.
    for idx in range(len(layers)):
        layer = layers[idx]
        layer_backprop_grads = paramGrads[idx]
    # Compute the numerical gradient for each parameter in the layer
        for p_idx, param in enumerate(layer.get_params_iter()):
            grad_backprop = layer_backprop_grads[p_idx] # + eps
            param += eps
            plusCost = layers[-1].get_cost(fprop(X_temp, layers)[-1], T_temp) # - eps
            param -= 2 * eps
            minCost = layers[-1].get_cost(fprop(X_temp, layers)[-1], T_temp) # reset param value
            param += eps # calculate numerical gradient
            grad_num = (plusCost - minCost) / (2 * eps)
        # Raise error if the numerical grade is not close to the backprop gradient
            secondGradText=''
            if layersFrom2nClassifier is not None:
                gradBackprop2ndClassifier = (paramGradsOf2nClassifier[idx])[p_idx] # + eps
                secondGradText='\n while the vectorized implementation as a backpropagation gradient of {:.6f}'.format(float(gradBackprop2ndClassifier))
            if not np.isclose(grad_num, grad_backprop,atol=5e-1):
                raise ValueError('WARNING : Numerical gradient of {:.6f} is not close to the backpropagation gradient of {:.6f}!'.format(float(grad_num), float(grad_backprop)),secondGradText)
            print('OK: Index {:} numerical gradient {:.6f} close to the backprop gradient of {:.6f}!'.format(p_idx, float(grad_num), float(grad_backprop)),secondGradText)
    
    print('No gradient errors found')
    return activations, paramGrads



def sgd(X_train, T_train, T_validation, X_validation, T_test, X_test, layers, nb_of_batches, max_nb_of_iterations, learning_rate, early_stop_threshold = 1e-3):
    # Create batches (X,Y) from the training set
    XT_batches = list(zip(
        np.array_split(X_train, nb_of_batches, axis=0),  # X samples
        np.array_split(T_train, nb_of_batches, axis=0)))  # Y targets

    # initalize some lists to store the cost for future analysis
    minibatch_costs = []
    training_costs = []
    validation_costs = []
    training_errors = []
    validation_errors = []
    test_errors = []
# Train for the maximum number of iterations
    for iteration in range(max_nb_of_iterations):
        for X, T in XT_batches: # For each minibatch sub-iteration
            activations = fprop(X, layers) # Get the activations
            minibatch_cost = layers[-1].get_cost(activations[-1], T) # Get cost
            minibatch_costs.append(minibatch_cost)
            param_grads = bprop(activations, T, layers) # Get the gradients
            update_params(layers, param_grads, learning_rate) # Update the parameters
        
    # Get full training cost for future analysis (plots)
        activations = fprop(X_train, layers) #     print('The accuracy on the training set is {:.2f}'.format(get_err_rate(T_train,activations)))
        training_errors.append(get_err_rate(T_train, activations)) #     print('The accuracy on the test set is {:.2f}'.format(get_err_rate(T_train,X_train,activations)))
        train_cost = layers[-1].get_cost(activations[-1], T_train)
        training_costs.append(train_cost) 
        # Get full validation cost
        activations = fprop(X_validation, layers)
        validation_cost = layers[-1].get_cost(activations[-1], T_validation)
        validation_costs.append(validation_cost) #     print('The accuracy on the validation set is {:.2f}'.format(get_err_rate(T_validation,activations)))
        validation_errors.append(get_err_rate(T_validation, activations)) #     print('The accuracy on the test set is {:.2f}'.format(get_err_rate(T_test,fprop(X_test, layers))))
        test_errors.append(get_err_rate(T_test, fprop(X_test, layers))) #     activations = fprop(X_test, layers)  # Get activation of test samples
    
        if len(validation_costs) > 3:
            # Stop training if the cost on the validation set doesn't decrease
            #  for 3 iterations
            if validation_costs[-1] >= validation_costs[-2] >= validation_costs[-3]:
                break
    #         if np.isclose(validation_costs[-1],  validation_costs[-2],atol=early_stop_threshold ):
    #             break
    nb_of_iterations = iteration + 1 # The number of iterations that have been executed
    return nb_of_iterations, minibatch_costs, training_costs, validation_costs, training_errors, validation_errors, test_errors, activations, layers



def trainClassifiersFromRange(X_train, X_validation, X_test, T_train, T_validation, T_test, thetaRange):
    lowest={'error':1}
    max_nb_of_iterations = 5
    classifierLayers=[]
    best_hyperparameters = []
    for nb_hidden_neurons in thetaRange['nb_hidden_neurons']:
        lowest={'error':1}
        for K_size_of_batch in thetaRange['K_size_of_batch']:
            for learning_rate in thetaRange['learning_rate']:
                nb_of_batches = int(X_train.shape[0] / K_size_of_batch) 
                layers = makeLayersWithVectorizedSoftmax(nb_hidden_neurons, X_train, T_train)
                nb_of_iterations, minibatch_costs, training_costs, validation_costs, training_errors, validation_errors, test_errors, activations, layers = sgd(X_train, T_train, T_validation, X_validation, T_test, X_test, layers, nb_of_batches, max_nb_of_iterations, learning_rate)
#                 print('\nrun with batch size: ', K_size_of_batch, '\tnumber of hidden neurons:', nb_hidden_neurons, '\tlearning rate:', learning_rate, '\tas an error rate of :', get_err_rate(T_test, fprop(X_test, layers)))
                curError=get_err_rate(T_test, fprop(X_test, layers))
                if curError<lowest['error']: lowest={'error':curError,'nb_hidden_neurons':nb_hidden_neurons,'K_size_of_batch':K_size_of_batch,'learning_rate':learning_rate,'activations':activations,'layers':layers, 'training_costs':training_costs, 'validation_costs':validation_costs, 'training_errors':training_errors, 'validation_errors':validation_errors, 'test_errors':test_errors,'nb_of_iterations':nb_of_iterations,'minibatch_costs':minibatch_costs,'nb_of_iterations':nb_of_iterations,'nb_of_batches':nb_of_batches}
        classifierLayers.append(lowest['layers'])
        best_hyperparameters.append(lowest)

    return classifierLayers, best_hyperparameters
#     classifierLayers[0]=lowest['layers']
#     print(lowest)
#     return nb_of_iterations, minibatch_costs, training_costs, validation_costs, training_errors, validation_errors, test_errors, activations, layers
#     return lowest['nb_of_iterations'], lowest['minibatch_costs'], lowest['training_costs'], lowest['validation_costs'], lowest['training_errors'], lowest['validation_errors'], lowest['test_errors'], lowest['nb_of_batches'],lowest['activations'], lowest['layers']


def trainOnRangeOfTheta(X_train, X_validation, X_test, T_train, T_validation, T_test, thetaRange,outFilePath='mnist_results.txt'):
    lowest={'error':1}
    max_nb_of_iterations = 5 
    with open(outFilePath,'w') as outFile:
        for nb_hidden_neurons in thetaRange['nb_hidden_neurons']:
            for K_size_of_batch in thetaRange['K_size_of_batch']:
                for learning_rate in thetaRange['learning_rate']:
                    nb_of_batches = int(X_train.shape[0] / K_size_of_batch)
                    layers = makeLayersWithVectorizedSoftmax(nb_hidden_neurons, X_train, T_train)
                    nb_of_iterations, minibatch_costs, training_costs, validation_costs, training_errors, validation_errors, test_errors, activations, sdgLayers = sgd(X_train, T_train, T_validation, X_validation, T_test, X_test, layers, nb_of_batches, max_nb_of_iterations, learning_rate)
                    print('\nrun with batch size: ', K_size_of_batch, '\tnumber of hidden neurons:', nb_hidden_neurons, '\tlearning rate:', learning_rate, '\tas an error rate of :', get_err_rate(T_test, fprop(X_test, sdgLayers)))
                    curError=get_err_rate(T_test, fprop(X_test, sdgLayers))
                    if curError<lowest['error']: lowest={'error':curError,'nb_hidden_neurons':nb_hidden_neurons,'K_size_of_batch':K_size_of_batch,'learning_rate':learning_rate,'activations':activations,'layers':sdgLayers, 'training_costs':training_costs, 'validation_costs':validation_costs, 'training_errors':training_errors, 'validation_errors':validation_errors, 'test_errors':test_errors,'nb_of_iterations':nb_of_iterations,'minibatch_costs':minibatch_costs,'nb_of_iterations':nb_of_iterations,'nb_of_batches':nb_of_batches}
            print()
            outFile.write('\nrun with batch size: '+str(K_size_of_batch)+'\tnumber of hidden neurons:'+str(nb_hidden_neurons)+ '\tlearning rate:'+str(learning_rate)+ '\tas an error rate of :'+str(get_err_rate(T_test, fprop(X_test, sdgLayers))))
        print(lowest)
    #     return nb_of_iterations, minibatch_costs, training_costs, validation_costs, training_errors, validation_errors, test_errors, activations, layers
    return lowest['nb_of_iterations'], lowest['minibatch_costs'], lowest['training_costs'], lowest['validation_costs'], lowest['training_errors'], lowest['validation_errors'], lowest['test_errors'], lowest['nb_of_batches'],lowest['activations'], lowest['layers']

def makeLayersWithVectorizedSoftmax(nb_hidden_neurons, X_train, T_train):
    layers = []
    layers.append(LinearLayer(X_train.shape[1], nb_hidden_neurons))
    layers.append(ReluLayer())
    layers.append(LinearLayer(nb_hidden_neurons, T_train.shape[1]))
    layers.append(VectorizedSoftmaxOutputLayer())
    return layers

def makeLayersWithNaiveSoftmax(nb_hidden_neurons, X_train, T_train):
    layers = []
    layers.append(LinearLayer(X_train.shape[1], nb_hidden_neurons))
    layers.append(ReluLayer())
    layers.append(LinearLayer(nb_hidden_neurons, T_train.shape[1]))
    layers.append(NaiveSoftmaxOutputLayer())
    return layers

#############################################################################################################################################
#############################################################################################################################################
#############################################################################################################################################

def  Q1_Q2():
    #Q1 & Q2 display gradient check of 1 sample from mood_data with an hidden layers with two nodes dh=2
    moonDataFilePath='2moons.txt'
    X_train, X_validation, X_test, T_train, T_validation, T_test = load_data_from_file(moonDataFilePath)
    nb_hidden_neurons = 2  # Number of neurons in the first hidden-layer
    layers = makeLayersWithNaiveSoftmax(nb_hidden_neurons, X_train, T_train)
     
    nb_samples_gradientcheck = 1 # Test the gradients on a subset of the data
    X_temp = X_train[0:nb_samples_gradientcheck,:]
    T_temp = T_train[0:nb_samples_gradientcheck,:]
    gradient_check(layers, X_temp, T_temp)
 
 
#############################################################################################################################################
def Q3_Q4():
    #Q3 & Q4 batch gradient check with loop 
    #Q4 display batch gradient check of 10 samples from mood_data with an hidden layers with two nodes dh=2
    moonDataFilePath='2moons.txt'
    X_train, X_validation, X_test, T_train, T_validation, T_test = load_data_from_file(moonDataFilePath)
    nb_hidden_neurons = 2  # Number of neurons in the first hidden-layer
    layers = makeLayersWithNaiveSoftmax(nb_hidden_neurons, X_train, T_train)
#     layers = makeLayersWithVectorizedSoftmax(nb_hidden_neurons, X_train, T_train)
    K_samples_gradientcheck = 10 # Test the gradients on a subset of the data
    X_temp = X_train[0:K_samples_gradientcheck,:]
    T_temp = T_train[0:K_samples_gradientcheck,:]
    gradient_check(layers, X_temp, T_temp)
#     for i in range(1,K_samples_gradientcheck):
#         gradient_check(layers, X_temp[i-1:i,:], T_temp[i-1:i,:])

#############################################################################################################################################
def Q5():
    #Q5 display decision region for different theta
    moonDataFilePath='2moons.txt'
    X_train, X_validation, X_test, T_train, T_validation, T_test = load_data_from_file(moonDataFilePath)
    thetaRange={'nb_hidden_neurons':np.linspace(2,50,5,dtype=int),'K_size_of_batch':np.linspace(1,50,10,dtype=int),'learning_rate':np.linspace(0.1,1,4)}
    classifiersLayers, hyperparameters =trainClassifiersFromRange(X_train, X_validation, X_test, T_train, T_validation, T_test, thetaRange)
    titles = []
    for i in range(0,len(hyperparameters)):
        buf = "hidden neurons = %d, batch size = %d, learning rate = %d" % (hyperparameters[i]['nb_hidden_neurons'],hyperparameters[i]['K_size_of_batch'],hyperparameters[i]['learning_rate'])
        titles.append(buf)
    multipleGridPlot(classifiersLayers, X_train, T_train, X_test, T_test, titles)


#############################################################################################################################################
def Q6_Q7():
    #6 explain difference between vectorized and loop implementation
    #Q7 display printout of gradient accuracy difference between moon_data with loop implementation vs vectorized with K=1 and K=10
    moonDataFilePath='2moons.txt'
    X_train, X_validation, X_test, T_train, T_validation, T_test = load_data_from_file(moonDataFilePath)

    for nb_samples_gradientcheck in [1,10]:
    #     nb_samples_gradientcheck = 10 # Test the gradients on a subset of the data
        X_temp = X_train[0:nb_samples_gradientcheck,:]
        T_temp = T_train[0:nb_samples_gradientcheck,:]
    
        nb_hidden_neurons = 2  # Number of neurons in the first hidden-layer
        layersWithNaiveSoftmax = makeLayersWithNaiveSoftmax(nb_hidden_neurons, X_train, T_train)
        layersWithVectorizedSoftmax = makeLayersWithVectorizedSoftmax(nb_hidden_neurons, X_train, T_train)
    
        #compare the two softmax implementations
        print('\nRunning Naive vs Vectorized gradient check') 
        gradient_check(layersWithNaiveSoftmax,X_temp, T_temp,layersWithVectorizedSoftmax)
    print('The naive implementation was modified by replacing loops by efficient numpy functions such as numpy.multiply as seen in the get_cost function.')


#############################################################################################################################################
#Q8 display time for moon_data with loop implementation vs vectorized with K=1 and K=10
def Q8():
    moonDataFilePath='2moons.txt'
    X_train, X_validation, X_test, T_train, T_validation, T_test = load_data_from_file(moonDataFilePath)

    msg=''
    
    for nb_samples_gradientcheck in [1,10]:
    #     nb_samples_gradientcheck = 10 # Test the gradients on a subset of the data
        X_temp = X_train[0:nb_samples_gradientcheck,:]
        T_temp = T_train[0:nb_samples_gradientcheck,:]
    
        nb_hidden_neurons = 2  # Number of neurons in the first hidden-layer
        layersWithNaiveSoftmax = makeLayersWithNaiveSoftmax(nb_hidden_neurons, X_train, T_train)
        
        print('\nRunning Naive gradient check') 
        t1 = time.clock()
        gradient_check(layersWithNaiveSoftmax, X_temp, T_temp)
        t2 = time.clock()
    
        print('\nRunning vectorized gradient check') 
        layersWithVectorizedSoftmax = makeLayersWithVectorizedSoftmax(nb_hidden_neurons, X_train, T_train)
        t3 = time.clock()
        gradient_check(layersWithVectorizedSoftmax, X_temp, T_temp)
        t4 = time.clock()
    
        #compare the two softmax implementations
        print('\nRunning Naive vs Vectorized gradient check') 
        gradient_check(layersWithNaiveSoftmax,X_temp, T_temp,layersWithVectorizedSoftmax)
        
#         print('\nThe naive implementation took {:.6f} seconds. While the vectorized implementation took : {:.6f} to predict {:d} samples points'.format(t2-t1,t4-t3, len(X_temp)))
        msg+='\n The naive implementation took {:.6f} seconds. While the vectorized implementation took : {:.6f} to predict {:d} samples points'.format(t2-t1,t4-t3, len(X_temp))
    print(msg)

#############################################################################################################################################
def Q9_Q10():
    #Q9 show print out of error and cost and save results to file
    picklefile='mnist.pkl.gz'
    #To load all samples
    X_train, X_validation, X_test, T_train, T_validation, T_test=load_data_with_pickle(picklefile)
    #To load subset of samples
#     nbOfTraingSampleToLoad=1000;
#     X_train, X_validation, X_test, T_train, T_validation, T_test=load_data_with_pickle(picklefile,nbOfTraingSampleToLoad)
    
#     rangeOfHyperParameters={'nb_hidden_neurons':range (50,256,78),'K_size_of_batch':range(50,400,100),'learning_rate':np.linspace(0.1,4,8)}
    rangeOfHyperParameters={'nb_hidden_neurons':[50],'K_size_of_batch':[50],'learning_rate':[0.1]}
    nb_of_iterations, minibatch_costs, training_costs, validation_costs, training_errors, validation_errors, test_errors, nb_of_batches, activations, layers = trainOnRangeOfTheta(X_train, X_validation, X_test, T_train, T_validation, T_test, rangeOfHyperParameters)
    #Q10 Mnist with good parameters
    plot_cost(nb_of_batches, nb_of_iterations, minibatch_costs, training_costs, validation_costs)
    plot_error(nb_of_batches, nb_of_iterations, training_errors, validation_errors, test_errors)
    plot_confusion_table(T_test, X_test, layers)


#############################################################################################################################################
#############################################################################################################################################
#Q1_Q2()
#Q3_Q4()
#Q5()
#Q6_Q7()
Q8()
#Q9_Q10()