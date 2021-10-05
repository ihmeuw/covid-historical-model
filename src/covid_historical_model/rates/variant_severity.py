from typing import Dict, List

import numpy as np

from covid_historical_model.utils.misc import get_random_state

RISK_RATIO = {'mean': 1.64, 'lower': 1.32, 'upper': 2.04}

def get_variant_severity_rr_dist(n_samples: int,
                                 risk_ratio: Dict[str, float] = RISK_RATIO) -> List[float]:
    mu = np.log(risk_ratio['mean'])
    sigma = (np.log(risk_ratio['upper']) - np.log(risk_ratio['lower'])) / 3.92
    rr = [np.exp(sample_rr(n, mu, sigma)) for n in range(n_samples)]
    
    return rr


def sample_rr(draw: int, mu: float, sigma: float,) -> float:
    random_state = get_random_state(f'rr_{draw}')
    
    return random_state.normal(mu, sigma)
