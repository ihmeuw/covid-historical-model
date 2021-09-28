from typing import List


from covid_historical_model.utils.misc import get_random_state


def get_cvi_dist(n_samples: int,
                 cvi_limits: List[int] = [0.3, 0.7]) -> List[float]:
    cvi = [sample_cvi(n, *cvi_limits) for n in range(n_samples)]
    
    return cvi


def sample_cvi(draw: int, lower: float, upper: float) -> float:
    random_state = get_random_state(f'chi_{draw}')
    
    return random_state.uniform(lower, upper)
