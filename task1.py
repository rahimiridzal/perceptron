import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pandas.plotting import scatter_matrix
import warnings
warnings.filterwarnings('ignore')

# this main.py file is divided into 3 parts
# part 1: input data representation
# part 2: define learning algorithm
# part 3: apply learning algorithm

# !!!part 1!!!

from sklearn.datasets import load_iris
iris = load_iris()

# if need more information uncomment below:
# print(iris["DESCR"])

# formatting the data using Panda's DataFrame
iris_data = pd.DataFrame(data=iris["data"], columns=iris["feature_names"])
iris_data["target"] = iris["target"]
color_dictionary = {1:"#FE010A", 2:"#2722DD", 3:"#1BE41D"}
colors = iris_data["target"].map(lambda x: color_dictionary.get(x + 1))
ax = scatter_matrix(iris_data, color=colors, alpha=0.8, figsize=(15,15), diagonal="hist")

# use the perceptron as binary classifier -> choose two classes from the three (Setosa and Versicolor)
# consider two features for better data visualisation in 2D plot (petal length and sepal length)

# [4] choose last features which is basically the label
setosa_label = iris_data.iloc[0:50,4].values
versicolor_label = iris_data.iloc[50:100,4].values
labels = np.concatenate((setosa_label, versicolor_label))

# change class label from 0 and 2 to -1 and 1
y = np.where(labels==0,-1,1)

# [0,2] choose 1st and 3rd features (petal length and sepal length)
X_setosa = iris_data.iloc[0:50,[0,2]]
X_versicolor = iris_data.iloc[50:100, [0,2]]
X = np.concatenate((X_setosa, X_versicolor))

# plot the features in 2D plot 
# sepal length vs petal length (x-axis vs y-axis)

plt.scatter(X[:50,0],X[:50,1],color="red",marker="o",label="Setosa")
plt.scatter(X[50:100,0], X[50:100,1],color="blue",marker="o",label="Versicolor")
plt.title("Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.show()


# !!!part 2!!!

# define functions needed to perform PLA

# return class label after each step
def predict(W, X):
    return np.where(net_input(W, X) >= 0, 1, -1)

# calculate net input
def net_input(W, X):
    return np.dot(X, W[1:]) + W[0]

# plot the boundary defined by the weights
# generate line from this eq: w0 + w1*x1 + w2*x2 = 0
def plot_boundary(W, X):
    x_vals = []
    y_vals = []

    for i in range(0,10):
        x_vals.append(i)
        result = (-1)*(W[1]*i + W[0])/W[2]
        y_vals.append(result)

# this part define the x-axis and y-axis style according to the data used (Iris data)
# can be edited depends on the features/labels used

    plt.scatter(X[:50,0],X[:50,1],color='red',marker='o',label='Setosa')
    plt.scatter(X[50:100,0],X[50:100,1],color='blue',marker='o',label='Versicolor') 
    plt.title("Sepal Length vs Petal Length")
    plt.xlabel('Sepal Length')    
    plt.ylabel('Petal Length')
    plt.legend(loc='upper left')
    plt.plot(x_vals,y_vals, 'black') #Plot using found values that form the line. 
    plt.show()

def plot_errors(error):
    plt.plot(range(1, len(error) + 1), error,marker='o')
    plt.xlabel('Iterations')
    plt.ylabel('Number of misclassifications')
    plt.show()

def fit(n_iter, X, y):
    W = np.zeros(1 + X.shape[1])
    errors_ = []        
    for i in range(n_iter):
        for j, (xi, target) in enumerate(zip(X,y)):
            update = target - predict(W, xi)
            W[1:] += update*xi
            W[0] += update

# errors on whole dataset 
        
        errors = 0
        for j, (xi, target) in enumerate(zip(X, y)):
            if predict(W, xi) != target:
                errors += 1
        errors_.append(errors) 
        i += 1
        print ("Iteration: ", i)
        print("Weights: ", W)

#When weights are updated we plot the boundary

        plot_boundary(W,X)
    plot_errors(errors_)


# !!!part 3!!!

n_iter=8
fit(n_iter, X, y)