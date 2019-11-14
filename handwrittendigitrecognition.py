from sklearn.datasets import fetch_openml
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

mnist = fetch_openml('mnist_784')
# print(mnist)

x,y = mnist['data'],mnist['target']


some_digit = x[36001]
some_digit_image = some_digit.reshape(28,28)
x_train, x_test = x[0:60000],x[60000:]
y_train, y_test = y[0:60000],y[6000:]


shuffle_index = np.random.permutation(60000)
x_train,y_train = x_train[shuffle_index],y_train[shuffle_index]


# Creating a two detector

y_train = y_train.astype(np.int8)
y_test = y_test.astype(np.int8)
y_train_2 = (y_train ==2)
y_test_2 = (y_test==2)



clf = LogisticRegression(tol = 0.1,solver = "lbfgs")
clf.fit(x_train,y_train_2)

clf.predict([some_digit])

# We have to do cross validation
a = cross_val_score(clf , x_train, y_train_2, cv = 3,scoring="accuracy")
c = a.mean()
print(c)
