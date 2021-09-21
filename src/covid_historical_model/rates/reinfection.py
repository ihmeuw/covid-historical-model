from pathlib import Path

import pandas as pd
import numpy as np

from covid_historical_model.etl.helpers import aggregate_data_from_md
from covid_historical_model.durations.durations import EXPOSURE_TO_DEATH, EXPOSURE_TO_SEROPOSITIVE

# def fill_non_covid_locs(escape_variant_prevalence: pd.Series,
#                         hierarchy: pd.DataFrame, gbd_hierarchy: pd.DataFrame,):
#     location_ids = gbd_hierarchy['location_id'].to_list()
#     covid_location_ids = hierarchy['location_id'].to_list()
#     location_ids = [l for l in location_ids if l not in covid_location_ids]
#     data_location_ids = escape_variant_prevalence.index.get_level_values(0).unique().to_list()
#     location_ids = [l for l in location_ids if l not in data_location_ids]
#     for location_id in location_ids:
#         parent_ids = gbd_hierarchy.loc[gbd_hierarchy['location_id'] == location_id, 'path_to_top_parent'].item()
#         parent_ids = [int(l) for l in reversed(parent_ids.split(','))][1:]
#         parent_ids = [l for l in parent_ids if l in data_location_ids]
#         if parent_ids:
#             escape_variant_prevalence = escape_variant_prevalence.append(
#                         pd.concat({location_id: escape_variant_prevalence.loc[parent_ids[0]]},
#                                   names=['location_id'])
#             )
    
#     return escape_variant_prevalence, location_ids


def add_repeat_infections(cross_variant_immunity: float,
                          escape_variant_prevalence: pd.Series,
                          daily_deaths: pd.Series, pred_ifr: pd.Series,
                          seroprevalence: pd.DataFrame,
                          hierarchy: pd.DataFrame,
                          gbd_hierarchy: pd.DataFrame,
                          population: pd.Series,
                          verbose: bool = True,
                          **kwargs) -> pd.DataFrame:
    infections = ((daily_deaths / pred_ifr)
                  .clip(1e-4, np.inf)
                  .dropna()
                  .rename('infections')
                  .reset_index())
    infections['date'] -= pd.Timedelta(days=EXPOSURE_TO_DEATH)
    infections = (infections
                  .set_index(['location_id', 'date'])
                  .loc[:, 'infections'])
    
    # # COULD use this to fill in, but that should come from variant modeler
    # escape_variant_prevalence, extra_locations = fill_non_covid_locs(escape_variant_prevalence,
    #                                                                  hierarchy, gbd_hierarchy)
    extra_locations = gbd_hierarchy.loc[gbd_hierarchy['most_detailed'] == 1, 'location_id'].to_list()
    extra_locations = [l for l in extra_locations if l not in hierarchy['location_id'].to_list()]
    
    escape_variant_prevalence = pd.concat([infections, escape_variant_prevalence], axis=1)  # borrow axis
    escape_variant_prevalence = escape_variant_prevalence['escape_variant_prevalence'].fillna(0)
    
    ancestral_infections = (infections * (1 - escape_variant_prevalence)).groupby(level=0).cumsum().dropna()
    repeat_infections = ((ancestral_infections / population) * (1 - cross_variant_immunity) * infections * escape_variant_prevalence).rename('infections')
    repeat_infections = repeat_infections.fillna(infections).dropna()
    
    obs_infections = infections.groupby(level=0).cumsum().dropna()
    first_infections = (infections - repeat_infections).groupby(level=0).cumsum().dropna()
    
    extra_obs_infections = obs_infections.reset_index()
    extra_obs_infections = (extra_obs_infections
                            .loc[extra_obs_infections['location_id'].isin(extra_locations)]
                            .reset_index(drop=True))
    obs_infections = aggregate_data_from_md(obs_infections.reset_index(), hierarchy, 'infections')
    obs_infections = obs_infections.append(extra_obs_infections)
    obs_infections = (obs_infections
                      .set_index(['location_id', 'date'])
                      .loc[:, 'infections'])
    extra_first_infections = first_infections.reset_index()
    extra_first_infections = (extra_first_infections
                              .loc[extra_first_infections['location_id'].isin(extra_locations)]
                              .reset_index(drop=True))
    first_infections = aggregate_data_from_md(first_infections.reset_index(), hierarchy, 'infections')
    first_infections = first_infections.append(extra_first_infections)
    first_infections = (first_infections
                        .set_index(['location_id', 'date'])
                        .loc[:, 'infections'])
    
    cumul_inflation_factor = (obs_infections / first_infections).rename('inflation_factor').dropna()
    daily_inflation_factor = (obs_infections.groupby(level=0).diff().replace(obs_infections).clip(1., np.inf) / \
                              first_infections.groupby(level=0).diff().replace(first_infections).clip(1., np.inf)
                             ).rename('inflation_factor').dropna().clip(1., np.inf)
    
    cumul_inflation_factor = cumul_inflation_factor.reset_index()
    cumul_inflation_factor['date'] += pd.Timedelta(days=EXPOSURE_TO_SEROPOSITIVE)
    daily_inflation_factor = daily_inflation_factor.reset_index()
    
    seroprevalence = seroprevalence.merge(cumul_inflation_factor, how='left')
    seroprevalence['inflation_factor'] = seroprevalence['inflation_factor'].fillna(1)
    
    cumul_inflation_factor['date'] -= pd.Timedelta(days=EXPOSURE_TO_SEROPOSITIVE)
    
    seroprevalence['seroprevalence'] *= seroprevalence['inflation_factor']
    
    del seroprevalence['inflation_factor']
    
    return cumul_inflation_factor, daily_inflation_factor, seroprevalence
