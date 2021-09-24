from typing import List

import numpy as np


def get_cvi_dist(n_samples: int,
                 cvi_limits: List[int] = [0.3, 0.7]):
    
    cvi = np.random.uniform(*cvi_limits,
                            size=n_samples)
    
    return cvi
    