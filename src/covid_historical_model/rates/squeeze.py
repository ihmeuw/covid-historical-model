import pandas as pd
import numpy as np

CEILING = 0.9

def squeeze(daily: pd.Series, rate: pd.Series,
            day_shift: int,
            population: pd.Series,
            reinfection_inflation_factor: pd.Series,
            vaccine_coverage: pd.DataFrame,
            ceiling: float = CEILING,) -> pd.Series:
    daily += 1e-4
    daily_infections = (daily / rate).dropna().rename('infections')
    daily_infections = daily_infections.reset_index()
    daily_infections['date'] -= pd.Timedelta(days=day_shift)
    daily_infections = daily_infections.set_index(['location_id', 'date']).loc[:, 'infections']
    cumul_infections = daily_infections.groupby(level=0).cumsum()
    cumul_infections = pd.concat([cumul_infections,
                                  reinfection_inflation_factor], axis=1)
    cumul_infections = cumul_infections.sort_index()
    cumul_infections['inflation_factor'] = (cumul_infections['inflation_factor']
                                            .groupby(level=0).apply(lambda x: x.fillna(method='ffill')))
    cumul_infections['inflation_factor'] = cumul_infections['inflation_factor'].fillna(1)
    cumul_infections['seroprevalence'] = cumul_infections['infections'] / cumul_infections['inflation_factor']
    seroprevalence = cumul_infections['seroprevalence'].dropna()
    cumul_infections = cumul_infections['infections'].dropna()
    
    max_cumul_infections = cumul_infections.groupby(level=0).max()
    max_seroprevalence = seroprevalence.groupby(level=0).max()
    
    ## don't worry about vaccinations for now...
    # vaccine_coverage = vaccine_coverage.join(daily, how='right')['cumulative_all_effective'].fillna(0)
    # vaccine_coverage = vaccine_coverage.groupby(level=0).max()
    # vaccine_coverage /= population
    # limits = ceiling * (1 - vaccine_coverage)
    # limits *= population
    
    limits = population * ceiling
    
    excess = (max_seroprevalence - limits).dropna().clip(0, np.inf)
    excess_scaling_factor = ((max_seroprevalence - excess) / max_seroprevalence).rename('scalar')
    excess_scaling_factor = excess_scaling_factor.fillna(1)
    
    rate = (rate / excess_scaling_factor).fillna(rate)
    
    adj_reinfection_inflation_factor = (cumul_infections * excess_scaling_factor) / (seroprevalence * excess_scaling_factor)
    
    return rate.dropna(), adj_reinfection_inflation_factor.fillna(1)
