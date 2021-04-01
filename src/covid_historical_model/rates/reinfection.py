from pathlib import Path

import pandas as pd

from covid_historical_model.etl.helpers import aggregate_data_from_md
from covid_historical_model.durations.durations import EXPOSURE_TO_DEATH, EXPOSURE_TO_SEROPOSITIVE

CROSS_VARIANT_IMMUNITY = 0.33


def add_repeat_infections(escape_variant_prevalence: pd.Series,
                          daily_deaths: pd.Series, pred_ifr: pd.Series,
                          seroprevalence: pd.DataFrame,
                          hierarchy: pd.DataFrame,
                          population: pd.Series,
                          cross_variant_immunity: float = CROSS_VARIANT_IMMUNITY,
                          verbose: bool = True) -> pd.DataFrame:
    infections = (daily_deaths / pred_ifr).dropna().rename('infections')
    infections = infections.reset_index()
    infections['date'] -= pd.Timedelta(days=EXPOSURE_TO_DEATH)
    infections = (infections
                  .set_index(['location_id', 'date'])
                  .loc[:, 'infections'])
    
    obs_infections = infections.groupby(level=0).cumsum().dropna()
    repeat_infections = ((obs_infections / population) * (1 - cross_variant_immunity) * infections * escape_variant_prevalence).rename('infections')
    repeat_infections = repeat_infections.fillna(infections).dropna()
    ancestral_infections = (infections - repeat_infections).groupby(level=0).cumsum().dropna()
    
    obs_infections = aggregate_data_from_md(obs_infections.reset_index(), hierarchy, 'infections')
    obs_infections = (obs_infections
                      .set_index(['location_id', 'date'])
                      .loc[:, 'infections'])
    ancestral_infections = aggregate_data_from_md(ancestral_infections.reset_index(), hierarchy, 'infections')
    ancestral_infections = (ancestral_infections
                            .set_index(['location_id', 'date'])
                            .loc[:, 'infections'])
    
    inflation_factor = (obs_infections / ancestral_infections).rename('inflation_factor').dropna()
    
    '''
    fig, ax = plt.subplots(4, figsize=(8, 8), sharex=True)
    ax[0].plot(escape_variant_prevalence.loc[196], color='red')
    ax[0].set_ylabel('P1 + B.1.351\nprevalence')

    ax[1].plot(infections.loc[196].rolling(window=7, min_periods=7, center=True).mean())
    ax[1].plot((infections - repeat_infections).loc[196].rolling(window=7, min_periods=7, center=True).mean())
    ax[1].plot(repeat_infections.loc[196].rolling(window=7, min_periods=7, center=True).mean())
    ax[1].set_ylabel('Daily infections')

    ax[2].plot(obs_infections.loc[196] / 58.56e6)
    ax[2].plot(ancestral_infections.loc[196] / 58.56e6)
    ax[2].set_ylabel('Cumulative infections (%)')

    ax[3].plot(inflation_factor.loc[196], color='purple')
    ax[3].set_ylabel('Inflation factor')

    ax[3].set_xlim(pd.Timestamp('2020-03-01'), pd.Timestamp('2021-03-01'))
    fig.show()
    '''
    
    inflation_factor = inflation_factor.reset_index()
    inflation_factor['date'] += pd.Timedelta(days=EXPOSURE_TO_SEROPOSITIVE)
    
    seroprevalence = seroprevalence.merge(inflation_factor, how='left')
    
    seroprevalence['seroprevalence'] *= seroprevalence['inflation_factor']
    
    del seroprevalence['inflation_factor']
    
    return inflation_factor, seroprevalence
