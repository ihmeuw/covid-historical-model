from typing import List
import hashlib

import numpy as np


def get_cvi_dist(n_samples: int,
                 cvi_limits: List[int] = [0.3, 0.7]) -> List[float]:
    cvi = [sample_cvi(n, *cvi_limits) for n in range(n_samples)]
    
    return cvi


def sample_cvi(draw: int, lower: float, upper: float) -> float:
    key = f'chi_{draw}'
    # 4294967295 == 2**32 - 1 which is the maximum allowable seed for a `numpy.random.RandomState`.
    seed = int(hashlib.sha1(key.encode('utf8')).hexdigest(), 16) % 4294967295
    random_state = np.random.RandomState(seed=seed)
    
    return random_state.uniform(lower, upper)
