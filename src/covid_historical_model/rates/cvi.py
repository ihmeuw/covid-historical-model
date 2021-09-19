import numpy as np

CROSS_VARIANT_IMMUNITY = [0.3, 0.7]


def get_cvi_dist(n_samples: int):
    cvi = np.random.uniform(*CROSS_VARIANT_IMMUNITY,
                            size=n_samples)
    
    return cvi
    