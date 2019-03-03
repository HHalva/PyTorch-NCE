import scipy as sp
from scipy import stats
import scipy.integrate as it
import numpy as np
import pdb
import sklearn as sk
from sklearn import datasets



p_noise = sp.stats.multivariate_normal(np.array([0,0,0,0,0]), np.eye(5))
cov = sk.datasets.make_spd_matrix(5)
p_data = sp.stats.multivariate_normal(np.array([0,0,0,0,0]), cov)

data = p_data.rvs(1000000)
noise = p_data.rvs(50000000)

def posterior_data(x):
    return np.log(p_data.pdf(x) / (p_data.pdf(x) + 50*p_noise.pdf(x)))

def posterior_noise(x):
    return np.log(50*p_noise.pdf(x) / (p_data.pdf(x) + 50*p_noise.pdf(x)))

print((np.mean(posterior_data(data)), np.mean(posterior_noise(noise))))




pdb.set_trace()

