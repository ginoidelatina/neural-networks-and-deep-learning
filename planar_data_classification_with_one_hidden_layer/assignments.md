# First import all the packages that you will need during this assignment.

- [numpy](https://www.numpy.org/) is the fundamental package for scientific computing with Python.
- [sklearn](http://scikit-learn.org/stable/) provides simple and efficient tools for data mining and data analysis.
- [matplotlib](http://matplotlib.org/) is a library for plotting graphs in Python.
- testCases provides some test examples to assess the correctness of your functions
- planar_utils provide various useful functions used in this assignment

# Exercise 1

How many training examples do you have? In addition, what is the `shape of the variables `X` and `Y`?

# Exercise 2 - layer_sizes

Define three variables:

- n_x: the size of the input layer
- n_h: the size of the hidden layer (set this to 4)
- n_y: the size of the output layer

**Hint**: Use shapes of X and Y to find n_x and n_y. Also, hard code the hidden layer size to be 4.

# Exercise 3 - initialize_parameters

**Implement the function `initialize_parameters()`.**

**Instructions:**

- Make sure your parameters' sizes are right. [Refer to the neural network figure above if you needed.](https://www.notion.so/Planar-Data-Classification-with-One-Hidden-Layer-05b9c67a23da4e4eb6e13ecd601b3573)
- You will initialize the weights matrices with random values.
    - Use: `np.random.randn(a,b) * 0.01` to randomly initialize a matrix of shape (a,b).
- You will initialize the bias vector as zeros.
    - Use: `np.zeros((a,b))` to initialize a matrix of shape (a, b) with zeros.

# Exercise 4 - forward_propagation

Implement `forward_propagation()` using the following equations:

- $Z^{[1]} =  W^{[1]} X + b^{[1]}$
- $A^{[1]} = \tanh(Z^{[1]})$
- $Z^{[2]} = W^{[2]} A^{[1]} + b^{[2]}$
- $\hat{Y} = A^{[2]} = \sigma(Z^{[2]})$

**Instructions:**

- Check the mathematical representation of your classifier in the figure above.
- Use the function `sigmoid()`. It's built into (imported) this notebook.
- Use the function `np.tanh()`. It's part of the numpy library.


Now that you've computed $A^{[2]}$ (in the Python variable `A2`), which contains $𝑎^{[2](𝑖)}$ for all examples, you can compute the cost function as follows:
$J = - \frac{1}{m} \sum\limits_{i = 1}^{m} \large{(} \small y^{(i)}\log\left(a^{[2] (i)}\right) + (1-y^{(i)})\log\left(1- a^{[2] (i)}\right) \large{)} \small$


# Exercise 5 - compute_cost

Implement `compute_cost()` to compute the value of the cost 𝐽.

**Instructions:**

* There are many ways to implement the cross-entropy loss. This is one way to implement one part of the equation without for loops:
  $-\sum\limits_{i = 1}^{m} y^{(i)}\log\(a^{[2] (i)})$

  `logprobs = np.multiply(np.log(A2),Y)`
  `cost =  - np.sum(logprobs)`

* Use that to build the whole expression of the cost function.

# Exercise 6 - backward_propagation
Implement the function `backward_propagation()`.

Instructions:

Backpropagation is usually the hardest (most mathematical) part in deep learning.  You'll want to use the six equations on the right of this slide, since you are building a vectorized implementation.

![grad_summary.png](https://github.com/ginoidelatina/neural-networks-and-deep-learning/blob/main/planar_data_classification_with_one_hidden_layer/images/grad_summary.png?raw=true)
**Tips:**

To compute `dZ1` you’ll need to compute $g^{[1]'}(Z^{[1]})$. Sinces $g^{[1]}(Z^{[1]})$ is the tanh activation function, 
if $A^{[1]} = g^{[1]}(Z^{[1]})$ then $g^{[1]'} (Z^{[1]}) = 1 - (A^{[1]})^2$. So you can compute $g^{[1]'} (Z^{[1]})$ using
`(1 - np.power(A1, 2))`.

# Exercise 7 - update_parameters

Implement the update rule. Use gradient descent. You have to use (dW1, db1, dW2, db2) in order to update (W1, b1, W2, b2).

General gradient descent rule: $\theta = \theta - \alpha \frac{\partial J }{ \partial \theta }$ where $\alpha$ is the learning hate and $\theta$ represents a parameter.

![sgd_bad.gif](https://github.com/ginoidelatina/neural-networks-and-deep-learning/blob/main/planar_data_classification_with_one_hidden_layer/images/sgd_bad.gif?raw=true)
![sgd.gif](https://github.com/ginoidelatina/neural-networks-and-deep-learning/blob/main/planar_data_classification_with_one_hidden_layer/images/sgd.gif?raw=true)

**Figure 2** : The gradient descent algorithm with a good learning rate (converging) and a bad learning rate (diverging). Images courtesy of Adam Harley.

**Hint**

- Use `copy.deepcopy(...)` when copying lists or dictionaries that are passed as parameters to functions. It avoids input parameters being modified within the function. In some scenarios, this could be inefficient, but it is required for grading purposes.

# Exercise 8 - nn_model
Build your neural network model in `nn_model()`.

**Instructions:**

The neural network model has to use the previous functions in the right order.

# Exercise 9 - predict

Predict with your model by building predict(). Use forward propagation to predict results.

Reminder:

$$
y_{prediction} = \mathbb 1 \text{{activation > 0.5}} = \begin{cases}
      1 & \text{if}\ activation > 0.5 \\
      0 & \text{otherwise}
    \end{cases}
$$

As an example, if you would like to set the entries of a matrix X to 0 and 1 based on a threshold you would do: `X_new = (X > threshold)`
