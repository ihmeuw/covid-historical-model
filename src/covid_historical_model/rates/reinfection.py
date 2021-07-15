from pathlib import Path

import pandas as pd

from covid_historical_model.etl.helpers import aggregate_data_from_md
from covid_historical_model.durations.durations import EXPOSURE_TO_DEATH, EXPOSURE_TO_SEROPOSITIVE

CROSS_VARIANT_IMMUNITY = 0.6


def generate_waning_dist(lower, upper, inflection=75, proportion_at_inflection=0.8, max_t=700, draws=1):
    immunity_reduction = np.random.uniform(lower, upper, draws)
    y_inflection = (1 - immunity_reduction * proportion_at_inflection)
    y_180 = (1 - immunity_reduction)
    m1 = (y_inflection - 1) / (inflection - 0)
    m2 = (y_180 - y_inflection) / (180 - inflection)
    t1 = np.repeat(np.arange(inflection).reshape(1, -1), draws, axis=0).T
    y1 = t1 * m1 + 1
    t2 = np.repeat(np.arange(inflection, max_t).reshape(1, -1), draws, axis=0).T
    y2 = t2 * m2 + (m1 - m2) * inflection + 1
    y = np.vstack([y1, y2])
    data = pd.DataFrame(y, columns=[f'draw_{i}' for i in range(draws)], index=pd.Index(np.arange(max_t), name='Days'))
    mean = data.mean(axis=1).rename('mean')
    # upper = data.quantile(.975, axis=1).rename('upper')
    # lower = data.quantile(.025, axis=1).rename('lower')
    # summary = pd.concat([mean, upper, lower], axis=1)
    
    return (1 - mean)  # data, summary


def divide_infections(escape_variant_prevalence: pd.Series,
                      daily_deaths: pd.Series, pred_ifr: pd.Series,
                      # seroprevalence: pd.DataFrame,
                      hierarchy: pd.DataFrame,
                      gbd_hierarchy: pd.DataFrame,
                      population: pd.Series,
                      cross_variant_immunity: float = CROSS_VARIANT_IMMUNITY,
                      verbose: bool = True) -> pd.DataFrame:
    infections = (daily_deaths / pred_ifr).dropna().rename('infections')
    infections = infections.reset_index()
    infections['date'] -= pd.Timedelta(days=EXPOSURE_TO_DEATH)
    infections = (infections
                  .set_index(['location_id', 'date'])
                  .loc[:, 'infections'])
    
    waning_rate = generate_waning_dist(0.05, 0.25, max_t=1000, draws=1000)
    
    ## MAY NEED THIS...?
    # extra_locations = gbd_hierarchy.loc[gbd_hierarchy['most_detailed'] == 1, 'location_id'].to_list()
    # extra_locations = [l for l in extra_locations if l not in hierarchy['location_id'].to_list()]
    
    escape_variant_prevalence = pd.concat([infections, escape_variant_prevalence], axis=1)  # borrow axis
    escape_variant_prevalence = escape_variant_prevalence['escape_variant_prevalence'].fillna(0)
    
    ancestral_infections = (infections * (1 - escape_variant_prevalence)).groupby(level=0).cumsum().dropna()
    repeat_variant_infections = ((ancestral_infections / population) * (1 - cross_variant_immunity) * infections * escape_variant_prevalence).rename('infections')
    repeat_variant_infections = repeat_variant_infections.fillna(infections).dropna()
    
    obs_infections = infections.groupby(level=0).cumsum().dropna()
    first_infections = (infections - repeat_variant_infections).groupby(level=0).cumsum().dropna()
    
    ## MAY NEED THIS...?
    # extra_obs_infections = obs_infections.reset_index()
    # extra_obs_infections = (extra_obs_infections
    #                         .loc[extra_obs_infections['location_id'].isin(extra_locations)]
    #                         .reset_index(drop=True))
    # obs_infections = aggregate_data_from_md(obs_infections.reset_index(), hierarchy, 'infections')
    # obs_infections = obs_infections.append(extra_obs_infections)
    # obs_infections = (obs_infections
    #                   .set_index(['location_id', 'date'])
    #                   .loc[:, 'infections'])
    # extra_first_infections = first_infections.reset_index()
    # extra_first_infections = (extra_first_infections
    #                           .loc[extra_first_infections['location_id'].isin(extra_locations)]
    #                           .reset_index(drop=True))
    # first_infections = aggregate_data_from_md(first_infections.reset_index(), hierarchy, 'infections')
    # first_infections = first_infections.append(extra_first_infections)
    # first_infections = (first_infections
    #                     .set_index(['location_id', 'date'])
    #                     .loc[:, 'infections'])
    
    
    '''
    fig, ax = plt.subplots(4, figsize=(8, 8), sharex=True)
    ax[0].plot(escape_variant_prevalence.loc[196], color='red')
    ax[0].set_ylabel('P1 + B.1.351\nprevalence')

    ax[1].plot(infections.loc[196].rolling(window=7, min_periods=7, center=True).mean())
    ax[1].plot((infections - repeat_infections).loc[196].rolling(window=7, min_periods=7, center=True).mean())
    ax[1].plot(repeat_infections.loc[196].rolling(window=7, min_periods=7, center=True).mean())
    ax[1].set_ylabel('Daily infections')

    ax[2].plot(obs_infections.loc[196] / 58.56e6)
    ax[2].plot(first_infections.loc[196] / 58.56e6)
    ax[2].set_ylabel('Cumulative infections (%)')

    ax[3].plot(inflation_factor.loc[196], color='purple')
    ax[3].set_ylabel('Inflation factor')

    ax[3].set_xlim(pd.Timestamp('2020-03-01'), pd.Timestamp('2021-03-01'))
    fig.show()
    '''
    
    return inflation_factor  # , seroprevalence
