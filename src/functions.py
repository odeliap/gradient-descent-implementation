# Libraries
#------------------------------------------------------------------------------
import numpy as np
import scipy.linalg as LA
import random


# Useful Linear / Polynomial Regression Functions
# ------------------------------------------------------------------------------
def A_mat(x, deg):
    """Create the matrix A part of the least squares problem.

    Args:
       x: vector of input data.
       deg: degree of the polynomial fit.

    Return:
        A: polynomial regression matrix with shape (x, deg+1)

    """

    # check the formatting of the input array
    # we need to format it to a "matrix" type
    if x.ndim == 1:
        x = np.array(x)
        x = x[:, None]

    # initialize A as column of ones
    A = np.ones((len(x), 1))

    # loop through degrees and add them as columns
    for exp in range(1, deg + 1):
        col = x**exp
        A = np.hstack((col, A))

    return A

def LLS_Solve(x, y, deg):
    """Find the vector w that solves the least squares regression.

    Args:
       x: vector of input data.
       y: vector of output data.
       deg: degree of the polynomial fit.

    Return:
       w: vector that solves least squares regression.

    """
    # get matrix A
    A = A_mat(x, deg)
    # get A transpose
    A_transpose = A.transpose()

    # get w = (A^T*A)^(-1)*A^T*y
    w = LA.inv(A_transpose.dot(A)).dot(A_transpose).dot(y)

    return w

def LLS_func(x, y, w, deg):
    """The linear least squares objective function.

    Args:
       x: vector of input data.
       y: vector of output data.
       w: vector of weights.
       deg: degree of the polynomial.

    Return:
       objective_func: the objective function for the linear least squares problem

    """
    # get matrix A
    A = A_mat(x, deg)
    # get norm of A times w minus y
    norm = LA.norm(A.dot(w) - y)
    # get number of datapoints
    # N = x.size
    N = x.size
    # get the objective function by squaring the norm
    objective_func = (norm**2)/N
    # return the objective function divided by number of datapoints N
    return objective_func


# Gradient Descent Functions
# ------------------------------------------------------------------------------

def LLS_deriv(x,y,w,deg):
    """Computes the derivative of the least squares cost function

    Args:
        x: data
        y: output data
        w: coefficient vector that solves least squares problem
        deg: degree of the least square model

    Return:
        deriv: derivative of the least squares cost function

    """

    # compute matrix A
    A = A_mat(x, deg)
    # compute A transpose
    A_transpose = np.transpose(A)

    # get number of datapoints
    # N = x.size
    N = x.size

    deriv = (2 * A_transpose.dot(A.dot(w) - y))/N

    return deriv


def gradient_descent(x, y, w, D, K):
    """"
    Implementation of gradient descent optimizing least squares cost function.
    Outputs plots of norm vs cost with respect to the number of iterations.

    Args:
        x: input data
        y: output data
        w: initial vector to optimize
        D: initial derivative vector
        K: tolerance signifying acceptable derivative norm for stopping the descent method

    Return:
        vector w which optimizes least squares cost function

    """
    # holds size (i.e. norm) of derivative vector at each iteration
    d_hist = []
    # holds cost of function at each iteration
    c_hist = []
    # hold number of iterations
    iterations = []
    count = 0

    deg = len(w) - 1

    while LA.norm(D) >= K:
        cost = LLS_func(x, y, w, deg)

        d_hist.append(LA.norm(D))
        c_hist.append(cost)

        count = count + 1
        iterations.append(count)

        eps = 1
        m = LA.norm(D) ** 2
        t = 0.5*m
        while LLS_func(x, y, w - eps * D, 1) > LLS_func(x, y, w, 1) - eps * t:
            eps *= 0.9

        w = w - (eps * D)

        D = LLS_deriv(x, y, w, deg)

    return w, d_hist, c_hist, iterations


def batch_gradient_descent(x, y, w, D, K, b):
    """"
    Implementation of mini-batch gradient descent optimizing least squares cost function.
    Outputs plots of norm vs cost with respect to the number of iterations.

    Args:
        x: input data
        y: output data
        w: initial vector to optimize
        D: initial derivative vector
        K: tolerance signifying acceptable derivative norm for stopping the descent method
        b: batch size

    Return:
        vector w which optimizes least squares cost function

    """
    # holds size (i.e. norm) of derivative vector at each iteration
    d_hist = []
    # holds cost of function at each iteration
    c_hist = []
    # hold number of iterations
    iterations = []
    count = 0

    deg = len(w) - 1

    # check that n < N
    if b < x.size:
        while LA.norm(D) >= K:
            # randomly get mini-batch of datapoints of size b
            elements = x.size
            random_elements = []
            # populate random_elements list
            while len(random_elements) != b:
                rand = random.randint(0, elements - 1)
                if rand not in random_elements:
                    random_elements.append(rand)

            count_x = 0
            count_y = 0

            x_subarr = []
            y_subarr = []

            # populate x_subarr with batch number of sampled points
            for element in x:
                if count_x in random_elements:
                    x_subarr.append(element)
                count_x += 1

            # populate y_subarr with corresponding batch number of sampled points
            for element in y:
                if count_y in random_elements:
                    y_subarr.append(element)
                count_y += 1

            x = np.array(x_subarr)
            y = np.array(y_subarr)

            cost = LLS_func(x, y, w, deg)

            d_hist.append(LA.norm(D))
            c_hist.append(cost)

            count = count + 1
            iterations.append(count)

            eps = 1
            m = LA.norm(D) ** 2
            t = 0.5 * m
            while LLS_func(x, y, w - eps * D, 1) > LLS_func(x, y, w, 1) - eps * t:
                eps *= 0.9

            w = w - (eps * D)

            D = LLS_deriv(x, y, w, deg)

            print(w)
            print(D)

    return w, d_hist, c_hist, iterations


# LASSO Regularization Functions
# ------------------------------------------------------------------------------

def soft_thresh(v, lam):
    """
    Perform the soft-thresholding operation of the vector v using parameter lam.

    Args:
        v: input vector (numpy array)
        lam: parameter lam

    Returns:
        s: vector output (numpy array)

    """
    s = []

    for x in v:
        if x > lam:
            s_i = x - lam
            s.append(s_i)
        elif abs(x) <= lam:
            s_i = 0
            s.append(s_i)
        elif x < -lam:
            s_i = x + lam
            s.append(s_i)

    return s


def LASSO_regression(x, y, w, D, K, lam):
    """
    Implementation of LASSO regression.

    Args:
        x: input data
        y: output data
        w: initial vector to optimize
        D: initial derivative vector
        K: tolerance signifying acceptable derivative norm for stopping the descent method

    Returns:
        vector w which optimizes least squares cost function

    """
    # set max number of iterations
    max_iterations = 10000

    # get number of datapoints
    # N = x.size
    N = x.size

    # adjust lambda to make it smaller
    lam = lam/N

    # holds size (i.e. norm) of derivative vector at each iteration
    d_hist = []
    # holds cost of function at each iteration
    c_hist = []
    # hold number of iterations
    iterations = []
    count = 0

    deg = len(w) - 1

    while (count < max_iterations) and (LA.norm(D) >= K):
        cost = LLS_func(x, y, w, deg)

        d_hist.append(LA.norm(D))
        c_hist.append(cost)

        count = count + 1
        iterations.append(count)

        v = w - (lam*D)
        w = soft_thresh(v, lam)

        D = LLS_deriv(x, y, w, deg)

    return w, d_hist, c_hist, iterations
