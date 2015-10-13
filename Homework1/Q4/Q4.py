__author__ = 'Ahmet'
from scipy.stats import norm
import numpy as np
from scipy.optimize import fmin_bfgs

def posterior(param, n, y):

    theta = param
    post = y*theta
    post -= n*np.log(1+np.exp(theta))
    post -= ((theta-mu)**2)/(2*sigma)

    return post

def calc_diff(theta, n, y):

    return posterior(theta, n, y) - np.log(norm.pdf(theta,mu,sigma))

calc_diff_min = lambda *args: -calc_diff(*args)

def reject(post, n, data, c):

    k = len(mode)

    # Draw samples from g(theta)
    theta = norm.rvs(mu, sigma, size=n)

    # Calculate probability under g(theta)
    gvals = np.array([np.log(norm.pdf(t, mu, sigma)) for t in theta])

    # Calculate probability under f(theta)
    fvals = np.array([post(t, data[0], data[1]) for t in theta])

    # Calculate acceptance probability
    p = np.exp(fvals - gvals + c)

    return theta[np.random.random(n) < p]

posterior_min = lambda *args: -posterior(*args)

mu = 0
sigma = 0.25
init_value = (0.1)

n = 5
y = 5
opt = fmin_bfgs(posterior_min, init_value,
          args=(n, y), full_output=True)
mode, var = opt[0], opt[3]

prob = 1 - norm.cdf(0,mode,var)
print(prob)

c_init = 0
print()
opt = fmin_bfgs(calc_diff_min,c_init,
                args=(5, 5),
                full_output=True)

print(opt[1])

prob = reject(posterior,1000,[n,y],opt[1])

counter = 0
for i in range(len(prob)):
    if prob[i] > 0:
        counter+=1
print(counter/prob.size)
