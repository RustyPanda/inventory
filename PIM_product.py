import numpy as np
import sklearn

def make_synthetic_deliveries(time, noise_level=1):
    base_demand = np.sin(time)
    noise = np.random.rand(len(base_demand)) * noise_level
    actual_demand = base_demand + noise
    return actual_demand


def run_product():
    past_time = np.arange(0, 6)
    noise_level = 1
    actual_demand = make_synthetic_deliveries(past_time, noise_level)
    future_time = past_time + past_time.max()

    sklearn.re

