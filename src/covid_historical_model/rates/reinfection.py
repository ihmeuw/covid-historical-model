from pathlib import Path

import pandas as pd

from covid_historical_model.etl import estimates
from covid_historical_model.durations.durations import EXPOSURE_TO_DEATH, EXPOSURE_TO_SEROPOSITIVE


def load_variant_prevalence(variant_scaleup_root: Path, verbose: bool = True):
    variant_prevalence = estimates.escape_variant_scaleup(variant_scaleup_root, verbose=verbose)
    
    return variant_prevalence


def add_repeat_infections(variant_scaleup_root: Path,
                          daily_deaths: pd.Series, pred_ifr: pd.Series,
                          seroprevalence: pd.DataFrame,
                          cross_variant_immunity: float = 0.33,) -> pd.DataFrame:
    variant_prevalence = load_variant_prevalence(variant_scaleup_root, verbose=verbose)
    ## assume these are indexed on exposure...?
    # variant_prevalence = variant_prevalence.reset_index()
    # variant_prevalence['date'] -= pd.Timedelta(days=EXPOSURE_TO_CASE)
    # variant_prevalence = (variant_prevalence
    #                       .set_index(['location_id', 'date'])
    #                       .loc[:, 'variant_prevalence'])

    
    infections = (daily_deaths / pred_ifr).dropna().rename('infections')
    infections = infections.reset_index()
    infections['date'] -= pd.Timedelta(days=EXPOSURE_TO_DEATH)
    infections = (infections
                  .set_index(['location_id', 'date'])
                  .loc[:, 'infections'])
    
    repeat_infections = ((1 - cross_variant_immunity) * infections * variant_prevalence).rename('infections').dropna()
    
    obs_infections = infections.groupby(level=0).cumsum().dropna()
    ancestral_infections = (infections - repeat_infections).groupby(level=0).cumsum().dropna()
    
    inflation_factor = (obs_infections / ancestral_infections).rename('inflation_factor').dropna()
    
    '''
    fig, ax = plt.subplots(4, figsize=(8, 8), sharex=True)
    ax[0].plot(variant_prevalence.loc[196], color='red')
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
    seroprevalence['seroprevalence_sub_vacccinated'] = seroprevalence['seroprevalence']
    
    seroprevalence['seroprevalence'] *= seroprevalence['inflation_factor']
    
    del seroprevalence['inflation_factor']
    
    return seroprevalence
