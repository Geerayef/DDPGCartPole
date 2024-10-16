import numpy as np


class OUNoise:
    def __init__(
        self,
        action_space_size,
        decay_period,
        mu=0.0,
        theta=0.1,
        max_sigma=0.3,
        min_sigma=0.05,
    ):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space_size
        self.low = -1
        self.high = 1
        self.reset()

    def reset(self):
        self.state = np.random.normal(self.mu, self.sigma, size=self.action_dim)

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(
            1.0, self.time / self.decay_period
        )
        return self.state

    def noise(self, action, time):
        self.time = time
        # return self.evolve_state()
        return np.clip(self.evolve_state(), self.low, self.high)
