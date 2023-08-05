import numpy as np

class GaussianNoise:
    def __init__(self, action_space, noise_decay, mu=0.0, sigma=1.0):
        self.mu = mu
        self.initial_sigma = sigma
        self.sigma = sigma
        self.decay_rate = noise_decay
        self.action_space = action_space
        self.ep_count = 0

    def noise(self):
        noise = np.random.normal(self.mu, self.sigma, self.action_space)
        noise = np.clip(noise, -1, 1)
        return noise

    def reset(self):
        self.ep_count += 1
        self.sigma = self.initial_sigma * ( 1 - ( self.ep_count / self.decay_rate ) )
        self.sigma = max(self.sigma, 0.01)

