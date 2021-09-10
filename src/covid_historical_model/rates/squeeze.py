import pandas as pd
import numpy as np

CEILING = 0.9

def squeeze(daily: pd.Series, rate: pd.Series,
            day_shift: int,
            population: pd.Series,
            daily_reinfection_inflation_factor: pd.Series,
            vaccine_coverage: pd.DataFrame,
            ceiling: float = CEILING,) -> pd.Series:
    daily += 1e-4
    daily_infections = (daily / rate).dropna().rename('infections')
    daily_infections = daily_infections.reset_index()
    daily_infections['date'] -= pd.Timedelta(days=day_shift)
    daily_infections = daily_infections.set_index(['location_id', 'date']).loc[:, 'infections']
    daily_infections = pd.concat([daily_infections,
                                  daily_reinfection_inflation_factor], axis=1)
    daily_infections = daily_infections.sort_index()
    daily_infections['inflation_factor'] = (daily_infections['inflation_factor']
                                            .groupby(level=0).apply(lambda x: x.fillna(method='ffill')))
    daily_infections['inflation_factor'] = daily_infections['inflation_factor'].fillna(1)
    daily_infections['seroprevalence'] = daily_infections['infections'] / daily_infections['inflation_factor']
    
    cumul_infections = (daily_infections['infections'].dropna()
                        .groupby(level=0).cumsum())
    seroprevalence = (daily_infections['seroprevalence'].dropna()
                      .groupby(level=0).cumsum())

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
    
    ## unnecessary, obvs
    # adj_reinfection_inflation_factor = (cumul_infections * excess_scaling_factor) / (seroprevalence * excess_scaling_factor)
    
    return rate.dropna()  # , adj_reinfection_inflation_factor.fillna(1)
