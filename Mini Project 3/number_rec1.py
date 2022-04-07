import matplotlib.pyplot as plt
import pylab as pl
import numpy as np
from number_gen import training_data, training_vals, view_set, generate_test_set

# method to initilize layer parameters
def init_layer_params(layer_dimens):

    # make a dictionary for the parameters
    params = {}
    L = len(layer_dimens)
    for i in range(1, L):
        params['W' + str(i)] = np.random.randn(layer_dimens[i], layer_dimens[i-1]) * 0.01
        params['b' + str(i)] = np.zeros((layer_dimens[i], 1))

        assert(params['W' + str(i)].shape == (layer_dimens[i], layer_dimens[i-1]))
        assert(params['b' + str(i)].shape == (layer_dimens[i], 1))
    return params

# forward propagation of the network without the activation function
# returns raw output of forward function from a single node
def lin_forward(A, W, b):
    Z = np.dot(W, A)+b

    # check if the shape matches
    assert(Z.shape == (W.shape[0], A.shape[1]))

    # Save the current values and return them
    cache = (A, W, b)

    return Z, cache

# define the activation functions and their derivatives
# Raw activation functions
def relu(Z):
    return Z*(Z>0)

def sigmoid(Z):
    return 1/(1+np.exp(-Z))

# Derivatives of the activation functions
def drelu_(Z):
    return 1. *(Z>0)

def dsigmoid_(Z):
    return sigmoid(Z)*(1-sigmoid(Z))

# Final activation functions with prev cache
def sigmoid_a(Z):
    return sigmoid(Z),Z

def relu_a(Z):
    return relu(Z),Z

# Backward propagation activation functions
def relu_b(dA, act_cache):
    return dA* drelu_(act_cache)

def sigmoid_b(dA,act_cache):
    return dA* dsigmoid_(act_cache)

# forward propagation with activation for a single layer
def act_forward_prop(A_prev, W, b, type):
    Z, lin_cache = lin_forward(A_prev, W, b)
    if type == "sigmoid":
        A, act_cache = sigmoid_a(Z)
    elif type =="relu":
        A, act_cache = relu_a(Z)
    
    assert(A.shape == (W.shape[0],A_prev.shape[1]))
    cache = (lin_cache, act_cache)

    return A, cache

# forward propagation for all layers
def layers_model_forward(X, params):
    caches = []
    A = X
    L = len(params) // 2   
    for l in range(1, L):
        A_prev = A 
        A, cache = act_forward_prop(A_prev,params['W'+str(l)],params['b'+str(l)],"relu")
        caches.append(cache)
    AL, cache = act_forward_prop(A,params['W'+str(L)],params['b'+str(L)],"sigmoid")
    caches.append(cache)
    return AL, caches

# sum of errors functions per epoch
def sse_epoch(AL, Y):
    m = Y.shape[1]
    sse = np.sum((Y-AL)**2)
    sse = np.squeeze(sse)
    assert(sse.shape == ())
    return sse

# backward propagation of a single node without activation
def lin_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = (1/m)*np.dot(dZ,A_prev.T)
    db = (1/m)*np.sum(dZ,axis=1,keepdims=True)
    dA_prev = np.dot(W.T,dZ)
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db

# backward propagation for a single layer
def act_backward_prop(dA, cache, type):
    lin_cache, act_cache = cache

    if type == "relu":
        dZ = relu_b(dA,act_cache)
        dA_prev, dW, db = lin_backward(dZ,lin_cache)
    elif type == "sigmoid":
        dZ = sigmoid_b(dA,act_cache)
        dA_prev, dW, db = lin_backward(dZ,lin_cache)
    
    return dA_prev, dW, db

# backward propagation for all layers
def layers_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    #Y = Y.reshape(AL.shape)
    
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    curr_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = act_backward_prop(dAL,curr_cache,"sigmoid")
    
    for l in reversed(range(L-1)):
        curr_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = act_backward_prop(grads["dA"+str(l+1)],curr_cache,"relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    return grads

# update the parameters of the model
def update_params(params, grads, learning_rate):
    L = len(params) // 2 
    for l in range(L):
        params["W" + str(l+1)] = params["W" + str(l+1)]-(learning_rate)*grads["dW"+str(l+1)] 
        params["b" + str(l+1)] = params["b" + str(l+1)]-(learning_rate)*grads["db"+str(l+1)]
    return params

def run_model(X, Y, layers_dims, learning_rate = 0.1, num_iterations = 3000):
    # np.random.seed(2)
    sses = [] 
    
    params = init_layer_params(layers_dims)
    
    for i in range(0, num_iterations):
        AL, caches = layers_model_forward(X, params)
        sse = sse_epoch(AL, Y)
        grads = layers_model_backward(AL, Y, caches)
        params = update_params(params, grads, learning_rate)
        sses.append(sse)
    # plot the sse
    # print(sse)
    # print(AL)
    # print(Y)
    plt.plot(np.squeeze(sses))
    plt.ylabel('SSE')
    plt.xlabel('Epoch')
    plt.title("Learning rate =" + str(learning_rate))
    # plt.show()

    
    return params

# Predictions given inputs
def predict(X,params):
    AL,caches=layers_model_forward(X,params)
    prediction=np.argmax(AL,axis=0)
    return prediction.reshape(1,prediction.shape[0])

def main():
        
    # reshape the training sets
    x = training_data.reshape((len(training_vals), -1))
    y = np.array(training_vals)
    # print(x)

    x_train = x.T
    y = y.reshape(y.shape[0], 1)
    y_train = y.T

    Y_train_=np.zeros((10,y_train.shape[1]))
    for i in range(y_train.shape[1]):
        Y_train_[y_train[0,i],i]=1
    # print(Y_train_)

    # N layer neural network
    layers_dims=[x_train.shape[0],10,10]

    params = run_model(x_train, Y_train_, layers_dims, num_iterations = 10000)
    # print(params)

    train_predictions = predict(x_train, params)
    print(y_train)

    print("Training Accuracy : "+ str(np.sum(train_predictions[0]==y_train[0])/y_train.shape[1] * 100)+" %")

    x_test, y_test = generate_test_set()

    test_predictions = predict(x_test, params)
    print(test_predictions[0])
    print(y_test)

    print("Testing Accuracy : "+ str(np.sum(test_predictions==y_test)/y_test.shape[0] * 100)+" %")

    test_x = x_test.T
    test_view = test_x.reshape((30, 9 , 5))

    view_set(test_view, test_predictions[0])

if __name__ == '__main__':
    main()
    