# All important package:
- [numpy](https://numpy.org/doc/1.20/) is the fundamental package for scientific computing with Python.
- [h5py](http://www.h5py.org/) is a common package to interact with a dataset that is stored on an H5 file.
- [matplotlib](http://matplotlib.org/) is a famous library to plot graphs in Python.
- [PIL](https://pillow.readthedocs.io/en/stable/) and [scipy](https://www.scipy.org/) are used here to test your model with your own picture at the end.


# Exercise 1

Find the values for:

- m_train (number of training examples)
- m_test (number of test examples)
- num_px (height = width of a training image)

# Exercise 2
Reshape the training and test data sets so that images of size 
(num_px, num_px, 3) are flattened into single vectors of shape 
(num_px  ∗  num_px  ∗  3, 1).

# Exercise 3 - sigmoid

Using your code from "Python Basics", implement `sigmoid()`. As you've seen in the figure above, you need to compute
$sigmoid(z) = \frac{1}{1 + e^{-z}}$ for $z = w^T x + b$ to make predictions. Use `np.exp()`.

C
You have to initialize w as a vector of zeros. If you don't know what numpy function to use, look up np.zeros() in the [Numpy library's documentation](https://numpy.org/doc/stable/reference/generated/numpy.zeros.html).

# Exercise 5 - propagate

Implement a function `propagate()` that computes the cost function and its gradient.

Forward Propagation:
* You get X
* You compute
  $A = \sigma(w^T X + b) = (a^{(1)}, a^{(2)}, ..., a^{(m-1)}, a^{(m)})$
* You calculate the cost function: 
  $J = -\frac{1}{m}\displaystyle\\sum_{i=1}^{m}(y^{(i)}\log(a^{(i)})+(1-y^{(i)})\log(1-a^{(i)}))$

Here are the two formulas you will be using:
$\frac{\partial J}{\partial w} = \frac{1}{m}X(A-Y)^T$
$\frac{\partial J}{\partial b} = \frac{1}{m} \displaystyle\\sum_{i=1}^m (a^{(i)}-y^{(i)})$

# Exercise 6

Write down the optimization function. The goals is to lean $w$ and $b$ by minimizing the cost function $J$. 
For a parameter $\theta$, the update rule is $\theta = \theta - \alpha \text{ } d\theta$, where $\alpha$ is the learning rate.

# Exercise 7 - predict

The previous function will output the learned w and b. We are able to use w and b to predict the labels for a dataset X. Implement the `predict()` function. 
There are two steps to computing predictions:
* Calculate $\hat{Y} = A = \sigma(w^T X + b)$
* Convert the entries of a into 0 (if activation <= 0.5) or 1 (if activation > 0.5), stores the predictions in a vector Y_prediction. If you wish, you can use an if/else 
* statement in a for loop (though there is also a way to vectorize this).

# Exercise 8 - model

* Y_prediction_test for your predictions on the test set
* Y_prediction_train for your predictions on the train set
* parameters (w, b), grads (dw, db), costs for the outputs of `optimize()`.
