import matplotlib.pyplot as plt
import pylab as pl
import numpy as np

# training data set in array form
training_data = np.array([
    [[0,1,1,1,0],
    [1,0,0,0,1],
    [1,0,0,0,1],
    [1,0,0,0,1],
    [1,0,0,0,1],
    [1,0,0,0,1],
    [1,0,0,0,1],
    [1,0,0,0,1],
    [0,1,1,1,0]],

    [[0,0,1,0,0],
    [0,1,1,0,0],
    [1,0,1,0,0],
    [0,0,1,0,0],
    [0,0,1,0,0],
    [0,0,1,0,0],
    [0,0,1,0,0],
    [0,0,1,0,0],
    [0,0,1,0,0]],

    [[0,1,1,1,0],
    [1,0,0,0,1],
    [0,0,0,0,1],
    [0,0,0,0,1],
    [0,0,0,1,0],
    [0,0,1,0,0],
    [0,1,0,0,0],
    [1,0,0,0,0],
    [1,1,1,1,1]],
    
    [[0,1,1,1,0],
    [1,0,0,0,1],
    [0,0,0,0,1],
    [0,0,0,0,1],
    [0,0,0,1,0],
    [0,0,0,0,1],
    [0,0,0,0,1],
    [1,0,0,0,1],
    [0,1,1,1,0]],
    
    [[0,0,0,1,0],
    [0,0,1,1,0],
    [0,0,1,1,0],
    [0,1,0,1,0],
    [0,1,0,1,0],
    [1,0,0,1,0],
    [1,1,1,1,1],
    [0,0,0,1,0],
    [0,0,0,1,0]],
    
    [[1,1,1,1,1],
    [1,0,0,0,0],
    [1,0,0,0,0],
    [1,1,1,1,0],
    [1,0,0,0,1],
    [0,0,0,0,1],
    [0,0,0,0,1],
    [1,0,0,0,1],
    [0,1,1,1,0]],
    
    [[0,1,1,1,0],
    [1,0,0,0,1],
    [1,0,0,0,0],
    [1,0,0,0,0],
    [1,1,1,1,0],
    [1,0,0,0,1],
    [1,0,0,0,1],
    [1,0,0,0,1],
    [0,1,1,1,0]],
    
    [[1,1,1,1,1],
    [0,0,0,0,1],
    [0,0,0,1,0],
    [0,0,0,1,0],
    [0,0,1,0,0],
    [0,0,1,0,0],
    [0,1,0,0,0],
    [0,1,0,0,0],
    [0,1,0,0,0]],
    
    [[0,1,1,1,0],
    [1,0,0,0,1],
    [1,0,0,0,1],
    [1,0,0,0,1],
    [0,1,1,1,0],
    [1,0,0,0,1],
    [1,0,0,0,1],
    [1,0,0,0,1],
    [0,1,1,1,0]],

    [[0,1,1,1,0],
    [1,0,0,0,1],
    [1,0,0,0,1],
    [1,0,0,0,1],
    [0,1,1,1,1],
    [0,0,0,0,1],
    [0,0,0,0,1],
    [1,0,0,0,1],
    [0,1,1,1,0]],
])

# Values of the training set
training_vals = [i for i in range(10)]

# View the set
def view_set(x, y):
    plt.figure(figsize=(10,len(x)))
    i=0
    for plot in x:
        plt.subplot(5,6, i+1)
        plt.axis('off')
        plt.imshow(plot, cmap='binary', vmin=0, vmax=1)
        i+=1
    plt.show()
    print(y)

view_set(training_data, training_vals)

# reshape the training sets
x = training_data.reshape((len(training_vals), -1))
y = np.array(training_vals)
print(x.shape)

x_train = x.T
y_train = y.reshape(1, y.shape[0])

Y_train_=np.zeros((10,y_train.shape[1]))
for i in range(y_train.shape[1]):
    Y_train_[y_train[0,i],i]=1

# method to generate test set
# change one from each row of a value
def generate_test_set():

    # train_copy = np.copy(training_data)
    test_data = []
    test_vals = []
    for i in range(len(training_data)):
        for j in range(3):
            copy = np.copy(training_data[i])
            for row in copy:
                col = np.random.randint(0, 5)
                if row[col] == 1:
                    row[col] = 0
                else:
                    switch = np.random.randint(0,5)
                    if switch == 1:
                        row[col] = 1
            test_data.append(copy)
            test_vals.append(i)
    
    view_set(test_data, test_vals)

generate_test_set()

# method to initilize layer parameters
def init_layer_params(layer_dimens):

    # make a dictionary for the parameters
    params = {}
    L = len(layer_dimens)
    for i in range(1, L):
        params['W' + str(i)] = np.random.randn(layer_dimens[i], layer_dimens[i-1] * 0.01)
        params['b' + str(i)] = np.zeros((layer_dimens[i], 1))

        assert(params['W' + str(i)].shape == (layer_dimens[i], layer_dimens[i-1]))
        assert(params['b' + str(i)].shape == (layer_dimens[i], 1))
    return params

# 