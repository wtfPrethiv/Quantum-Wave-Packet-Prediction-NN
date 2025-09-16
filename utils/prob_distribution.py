import numpy as np

hbar = 1.0
m = 1.0
x0 = 0.0        # initial position
p0 = 5.0        # initial momentum

def electron_prob_density(sigma_t, x0, x, p0, t):
    
    norm = 1.0 / (2 * np.pie * np.square(sigma_t))
    
    return norm * np.exp(- (x - x0 - np.square((p0 / m) * t)) / (2 * np.square(sigma_t)))