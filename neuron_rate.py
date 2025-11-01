import numpy as np

def phi_relu(x):  # smooth enough
    return np.maximum(0.0, x)

class RateUnit:
    def __init__(self, tau=10.0):
        self.tau = float(tau)
        self.r = 0.0
    def step(self, dt, drive):
        self.r += dt * (-self.r + phi_relu(drive)) / self.tau
        return self.r
