# Libraries
#------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as LA


from functions import gradient_descent, batch_gradient_descent, LASSO_regression, LLS_Solve

# Gradient Descent with Athens Temperature Data
#------------------------------------------------------------------------------

# load athens data into pandas dataframe
athens_data = pd.read_csv('../datasets/athens_ww2_weather.csv')

# get min temp from athens dataframe
min_temp = athens_data['MinTemp']
# get max temp from athens dataframe
max_temp = athens_data['MaxTemp']

# set initial vector w, initial derivative vector D, and K (stopping criteria)
w = np.array([100, -100])
D = np.array([-1, 1])
K = 0.01

# set degree
degree = len(w) - 1

# get results from gradient descent method
w_optimized, d_hist, c_hist, iterations = gradient_descent(min_temp, max_temp, w, D, K)
print('Gradient Descent Solution: {}'.format(w_optimized))

# plot results: norms and costs
fig, axs = plt.subplots(2, figsize=(10, 8))
fig.tight_layout()

axs[0].plot(iterations, d_hist)
axs[0].set_title('Size by iteration for gradient descent')
axs[0].set_ylim([0, 500])

axs[1].plot(iterations, c_hist)
axs[1].set_title('Cost by iteration for gradient descent')
axs[1].set_xlim([1, 5])

plt.show()


# Batch Gradient Descent with Athens Temperature Data
#------------------------------------------------------------------------------

# set batch sizes
batch_sizes = [5, 10, 25, 50]

# initialize lists to store each batch norms, costs, and iterations
consolidated_batch_d_hists = []
consolidated_batch_c_hists = []
consolidated_batch_iterations = []

# perform gradient descent on batches of each size
for size in batch_sizes:
    b = size
    # get results from batch gradient descent
    batch_w_optimized, batch_d_hist, batch_c_hist, batch_iterations = batch_gradient_descent(min_temp, max_temp, w, D, K, b)
    print('Batch {} Gradient Descent Solution: {}'.format(size, batch_w_optimized))

    # add batch results to consolidated results list
    consolidated_batch_d_hists.append(batch_d_hist)
    consolidated_batch_c_hists.append(batch_c_hist)
    consolidated_batch_iterations.append(batch_iterations)

    # plot individual batch run norms and costs
    fig, axs = plt.subplots(2, figsize=(10, 8))
    fig.tight_layout()
    axs[0].plot(batch_iterations, batch_d_hist)
    axs[0].set_title('Size by iteration for batch size of {}'.format(size))
    axs[0].set_ylim([0, 500])
    axs[1].plot(batch_iterations, batch_c_hist)
    axs[1].set_title('Cost by iteration for batch size of {}'.format(size))
    axs[1].set_xlim([1, 5])
    plt.show()

# plot individual batch results on consolidated graph
fig, axs = plt.subplots(2, figsize=(10, 8))
fig.tight_layout()
axs[0].set_title('Consolidated batch size by iteration')
axs[1].set_title('Consolidated batch cost by iteration')
for i in range(0, 4):
    batch_d_hist = consolidated_batch_d_hists[i]
    batch_c_hist = consolidated_batch_c_hists[i]
    batch_iterations = consolidated_batch_iterations[i]
    axs[0].plot(batch_iterations, batch_d_hist, alpha=0.3, label='batch size: {}'.format(batch_sizes[i]))
    axs[0].set_ylim([0, 500])
    axs[0].legend()
    axs[1].plot(batch_iterations, batch_c_hist, alpha=0.3, label='batch size: {}'.format(batch_sizes[i]))
    axs[1].set_xlim([1, 5])
    axs[1].legend()
plt.show()


# Stochastic Gradient Descent with Athens Data
#------------------------------------------------------------------------------

# get results from stochastic gradient descent method
# get results from stochastic gradient descent method
# stochastic is the special case where batch size is 1
stochastic_w_optimized, stochastic_d_hist, stochastic_c_hist, stochastic_iterations = batch_gradient_descent(min_temp, max_temp, w, D, K, 1)
print('Stochastic Gradient Descent Solution: {}'.format(stochastic_w_optimized))

# plot norms and costs by iteration
fig, axs = plt.subplots(2, figsize=(10, 8))
fig.tight_layout()
axs[0].plot(stochastic_iterations, stochastic_d_hist)
axs[0].set_title('Size by iteration for stochastic gradient descent')
axs[0].set_ylim([0, 500])
axs[1].plot(stochastic_iterations, stochastic_c_hist)
axs[1].set_title('Cost by iteration for stochastic gradient descent')
plt.show()


# Aggregated Curves
#------------------------------------------------------------------------------

fig, axs = plt.subplots(2, figsize=(10, 8))
fig.tight_layout()

# plot gradient descent method norms
axs[0].plot(iterations, d_hist, alpha=0.3, label='gradient descent')
# plot stochastic gradient descent norms
axs[0].plot(stochastic_iterations, stochastic_d_hist, label='stochastic gradient descent')

for i in range(0, 4):
    batch_d_hist = consolidated_batch_d_hists[i]
    batch_c_hist = consolidated_batch_c_hists[i]
    batch_iterations = consolidated_batch_iterations[i]
    # plot batch gradient descent norms
    axs[0].plot(batch_iterations, batch_d_hist, alpha=0.3, label='batch size: {}'.format(batch_sizes[i]))
    # plot batch gradient descent costs
    axs[1].plot(batch_iterations, batch_c_hist, alpha=0.3, label='batch size: {}'.format(batch_sizes[i]))

# plot gradient descent costs
axs[1].plot(iterations, c_hist, label='gradient descent')
# plot stochastic gradient descent costs
axs[1].plot(stochastic_iterations, stochastic_c_hist, label='stochastic gradient descent')

# set plot configurations
axs[0].set_title('Aggregated size by iteration')
axs[0].set_ylim([0, 500])
axs[0].legend()

axs[1].set_title('Aggregated cost by iteration')
axs[1].set_xlim([1, 5])
axs[1].legend()

plt.show()


# LASSO Regularization with *Athens Data
#------------------------------------------------------------------------------

# set initial lambda vaue to zero
lam = 0.1

# initiate lists to hold optimal w vectors and exact solutions
optimal_ws = []
exact_solutions = []

# load athens data into pandas dataframe
athens_data = pd.read_csv('../datasets/athens_ww2_weather.csv')

# get min temp from athens dataframe
min_temp = athens_data['MinTemp']
# get max temp from athens dataframe
max_temp = athens_data['MaxTemp']

# set initial vector w, initial derivative vector D, and K (stopping criteria)
w = np.array([100, -100])
D = np.array([-1, 1])
K = 0.01

# set degree
degree = len(w) - 1

# initialize list to hold lambda values
lambdas = []

# initialize lists to hold l1 and l2 norms
l1_norms = []
l2_norms = []

# perform LASSO regression with values of lam ranging from 0.25 up to 5, spacing them in increments of 0.25
while lam <= 1.5:
    # add lam to lambdas list
    lambdas.append(lam)
    # get results from lasso regression method
    lasso_w, lasso_d_hist, lasso_c_hist, lass_iterations = LASSO_regression(min_temp, max_temp, w, D, K, lam)
    print('LASSO Regularization Solution for lambda = {}: {}'.format(lam, lasso_w))
    # save optimal w to optimal_ws list
    optimal_ws.append(lasso_w)
    # save exact solution to exact_solutions list
    exact_w = LLS_Solve(min_temp, max_temp, degree)
    # add l1 and l2 norm to respective lists
    L1 = LA.norm(lasso_w, ord = 1)
    L2 = LA.norm(lasso_w)
    l1_norms.append(L1)
    l2_norms.append(L2)
    # increment lam by 0.25
    lam = lam + 0.1

fig, axs = plt.subplots(figsize=(10, 8))
fig.tight_layout()

# plot norm of optimal w as a function of lambda
axs.plot(lambdas, l1_norms, alpha=0.3, label='l1 norms')
axs.plot(lambdas, l2_norms, alpha=0.3, label='l2 norms')

# set axes labels
axs.set_title('Norms as a function of lambda')
axs.set_xlabel('Lambda')
axs.set_ylabel('Norm')

axs.legend()
plt.show()


fig, axs = plt.subplots(2, figsize=(10, 8))
fig.tight_layout(pad=3.5)

constants = []
first_order = []

for vector in optimal_ws:
    coeff1 = vector[0]
    coeff2 = vector[1]
    constants.append(coeff2)
    first_order.append(coeff1)


axs[0].plot(lambdas, constants)
axs[1].plot(lambdas, first_order)

axs[0].set_title('Constant coefficients as a function of lambda')
axs[0].set_ylabel('Constant coefficient')
axs[0].set_xlabel('Lambda')
axs[1].set_title('First order coefficients as a function of lambda')
axs[1].set_ylabel('First order coefficient')
axs[1].set_xlabel('Lambda')

plt.show()

# There is a very discernible trend of the sizes of coefficients for increasing values of lambda
# The constant is trending towards 5 (or close to it) and the first order coefficient is trending towards 1.25