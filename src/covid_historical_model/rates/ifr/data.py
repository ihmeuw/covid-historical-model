from pathlib import Path
from typing import Dict, List
import itertools

import pandas as pd
import numpy as np

from covid_historical_model.etl import db, model_inputs, estimates
from covid_historical_model.durations.durations import SERO_TO_DEATH


def load_input_data(model_inputs_root: Path, age_pattern_root: Path,
                    seroprevalence: pd.DataFrame,
                    verbose: bool = True) -> Dict:
    # load data
    hierarchy = model_inputs.hierarchy(model_inputs_root)
    population = model_inputs.population(model_inputs_root)
    age_spec_population = model_inputs.population(model_inputs_root, by_age=True)
    cumulative_deaths, daily_deaths = model_inputs.reported_epi(model_inputs_root, 'deaths')
    sero_age_pattern = estimates.seroprevalence_age_pattern(age_pattern_root)
    ifr_age_pattern = estimates.ifr_age_pattern(age_pattern_root)
    covariates = [db.obesity(hierarchy)]
    
    return {'cumulative_deaths': cumulative_deaths,
            'daily_deaths': daily_deaths,
            'seroprevalence': seroprevalence,
            'covariates': covariates,
            'sero_age_pattern': sero_age_pattern,
            'ifr_age_pattern': ifr_age_pattern,
            'age_spec_population': age_spec_population,
            'hierarchy': hierarchy,
            'population': population,}


def create_model_data(cumulative_deaths: pd.Series, daily_deaths: pd.Series,
                      seroprevalence: pd.DataFrame,
                      covariates: List[pd.Series],
                      hierarchy: pd.DataFrame, population: pd.Series,
                      day_0: pd.Timestamp,
                      **kwargs) -> pd.DataFrame:
    ifr_data = seroprevalence.loc[seroprevalence['is_outlier'] == 0].copy()
    ifr_data['date'] += pd.Timedelta(days=SERO_TO_DEATH)
    ifr_data = (ifr_data
                .set_index(['location_id', 'date'])
                .loc[:, 'seroprevalence'])
    ifr_data = ((cumulative_deaths / (ifr_data * population))
                .dropna()
                .rename('ifr'))

    # get mean day of death int
    loc_dates = ifr_data.index.drop_duplicates().to_list()
    time = []
    for location_id, survey_end_date in loc_dates:
        locdeaths = daily_deaths.loc[location_id]
        locdeaths = locdeaths.reset_index()
        locdeaths = locdeaths.loc[locdeaths['date'] <= survey_end_date]
        locdeaths['t'] = (locdeaths['date'] - day_0).dt.days
        t = np.average(locdeaths['t'], weights=locdeaths['daily_deaths'] + 1e-4)
        mean_death_date = locdeaths.loc[locdeaths['t'] == int(np.round(t)), 'date'].item()
        time.append(
            pd.DataFrame(
                {'t':t, 'mean_death_date':mean_death_date},
                index=pd.MultiIndex.from_arrays([[location_id], [survey_end_date]],
                                                names=('location_id', 'date')),)
        )
    time = pd.concat(time)

    # add time
    model_data = time.join(ifr_data, how='outer')
    
    # add covariates
    for covariate in covariates:
        model_data = model_data.join(covariate, how='outer')
            
    return model_data.reset_index()


def create_pred_data(hierarchy: pd.DataFrame, population: pd.Series,
                     covariates: List[pd.Series],
                     pred_start_date: pd.Timestamp, pred_end_date: pd.Timestamp,
                     day_0: pd.Timestamp,
                     **kwargs):
    pred_data = pd.DataFrame(list(itertools.product(hierarchy['location_id'].to_list(),
                                               list(pd.date_range(pred_start_date, pred_end_date)))),
                         columns=['location_id', 'date'])
    pred_data['intercept'] = 1
    pred_data['t'] = (pred_data['date'] - day_0).dt.days
    pred_data = pred_data.set_index(['location_id', 'date'])
    
    for covariate in covariates:
        pred_data = pred_data.join(covariate, how='outer')
    
    return pred_data.reset_index()
