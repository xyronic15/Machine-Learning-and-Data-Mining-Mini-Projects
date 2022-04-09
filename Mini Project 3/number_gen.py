import matplotlib.pyplot as plt
import pylab as pl
import numpy as np

# training data set in array form
training_data = np.array([
    [[0,0.9,1,1,0],
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
        plt.title(y[i])
        i+=1
    plt.show()
    # print(y)

# view_set(training_data, training_vals)

# method to generate test set
# change one from each row of a value
def generate_test_set():

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
    
    # view_set(test_data, test_vals)

    test_data = np.array(test_data)
    test_data = test_data.reshape(30, -1)
    x_test = test_data.T

    test_vals = np.array(test_vals)
    
    return x_test, test_vals