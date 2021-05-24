from pathlib import Path
from typing import Dict, List
from loguru import logger
import itertools

import pandas as pd
import numpy as np

from covid_historical_model.etl import model_inputs, estimates
from covid_historical_model.durations.durations import (
    EXPOSURE_TO_DEATH, EXPOSURE_TO_CASE, PCR_TO_SERO, EXPOSURE_TO_SEROPOSITIVE
)


def load_input_data(model_inputs_root: Path, excess_mortality: bool, testing_root: Path,
                    shared: Dict, seroprevalence: pd.DataFrame, vaccine_coverage: pd.DataFrame,
                    verbose: bool = True) -> Dict:
    # load data
    cumulative_cases, daily_cases = model_inputs.reported_epi(
        model_inputs_root, 'cases', shared['hierarchy'], shared['gbd_hierarchy']
    )
    _, daily_deaths = model_inputs.reported_epi(
        model_inputs_root, 'deaths', shared['hierarchy'], shared['gbd_hierarchy'], excess_mortality
    )
    testing_capacity = estimates.testing(testing_root)['testing_capacity']

    covariates = []
    input_data = {
        'cumulative_cases': cumulative_cases,
        'daily_cases': daily_cases,
        'daily_deaths': daily_deaths,
        'seroprevalence': seroprevalence,
        'vaccine_coverage': vaccine_coverage,
        'testing_capacity': testing_capacity,
        'covariates': covariates,
    }
    input_data.update(shared)
    
    
    return input_data


def create_infections_from_deaths(daily_deaths: pd.Series, pred_ifr: pd.Series,) -> pd.Series:
    daily_deaths = (daily_deaths
                    .reset_index()
                    .groupby('location_id')
                    .apply(lambda x: pd.Series(x['daily_deaths'].rolling(window=7, min_periods=7, center=True).mean().values,
                                            index=x['date']))
                    .dropna())

    infections = (daily_deaths / pred_ifr).rename('infections').dropna().sort_index().reset_index()
    infections['date'] -= pd.Timedelta(days=EXPOSURE_TO_DEATH)
    infections = infections.set_index(['location_id', 'date'])
            
    return infections


def get_infection_weighted_avg_testing(infections: pd.Series, testing_capacity: pd.Series,
                                       verbose: bool = True) -> pd.Series:
    infwavg_data = pd.concat([testing_capacity.rename('testing_capacity'), infections.rename('infections')], axis=1)
    infwavg_data = infwavg_data.loc[infwavg_data['testing_capacity'].notnull()]
    infwavg_data['infections'] = infwavg_data['infections'].fillna(method='bfill')
    if infwavg_data.isnull().any().any():
        if verbose:
            logger.warning(f"Missing tail infections for location_id {infwavg_data.reset_index()['location_id'].unique().item()}.")
        infwavg_data['infections'] = infwavg_data['infections'].fillna(method='ffill')
    if not infwavg_data.empty:
        infwavg_testing_capacity = np.average(infwavg_data['testing_capacity'], weights=(infwavg_data['infections'] + 1))
        return pd.Series(infwavg_testing_capacity,
                         name='infwavg_testing_capacity',
                         index=infwavg_data.index[[-1]])
    else:
        return pd.Series(np.array([]),
                         name='infwavg_testing_capacity',
                         index=infwavg_data.index)
    

def create_model_data(cumulative_cases: pd.Series,
                      daily_cases: pd.Series,
                      seroprevalence: pd.DataFrame,
                      testing_capacity: pd.Series,
                      daily_deaths: pd.Series, pred_ifr: pd.Series,
                      covariates: List,
                      hierarchy: pd.DataFrame, population: pd.Series,
                      verbose: bool = True,
                      **kwargs):
    idr_data = seroprevalence.loc[seroprevalence['is_outlier'] == 0].copy()
    idr_data['date'] -= pd.Timedelta(days=PCR_TO_SERO)
    idr_data = (idr_data
                .set_index(['location_id', 'date'])
                .loc[:, 'seroprevalence'])
    idr_data = ((cumulative_cases / (idr_data * population))
                .dropna()
                .rename('idr'))
    
    infections = create_infections_from_deaths(daily_deaths, pred_ifr)
    infections = infections.reset_index()
    
    testing_capacity = testing_capacity.reset_index()
    testing_capacity['date'] -= pd.Timedelta(days=EXPOSURE_TO_CASE)
    sero_location_dates = seroprevalence[['location_id', 'date']].drop_duplicates()
    sero_location_dates = list(zip(sero_location_dates['location_id'], sero_location_dates['date']))
    infwavg_testing_capacity = []
    for location_id, date in sero_location_dates:
        infwavg_testing_capacity.append(
            get_infection_weighted_avg_testing(
                (infections
                 .loc[(infections['location_id'] == location_id) &
                      (infections['date'] <= (date - pd.Timedelta(days=EXPOSURE_TO_SEROPOSITIVE)))]
                 .set_index(['location_id', 'date'])
                 .loc[:, 'infections']),
                (testing_capacity
                 .loc[(testing_capacity['location_id'] == location_id) &
                      (testing_capacity['date'] <= (date - pd.Timedelta(days=EXPOSURE_TO_SEROPOSITIVE)))]
                 .set_index(['location_id', 'date'])
                 .loc[:, 'testing_capacity']),
                verbose,
            )
        )
    infwavg_testing_capacity = pd.concat(infwavg_testing_capacity,
                                         names='infwavg_testing_capacity').reset_index()
    infwavg_testing_capacity['date'] += pd.Timedelta(days=EXPOSURE_TO_CASE)
    infwavg_testing_capacity = (infwavg_testing_capacity
                                .set_index(['location_id', 'date'])
                                .loc[:, 'infwavg_testing_capacity'])
    infections = infections.set_index(['location_id', 'date'])
    
    log_infwavg_testing_rate_capacity= (np.log(infwavg_testing_capacity / population)
                                        .rename('log_infwavg_testing_rate_capacity'))
    del infwavg_testing_capacity

    # add testing capacity
    model_data = log_infwavg_testing_rate_capacity.to_frame().join(idr_data, how='outer')
    
    # add covariates
    for covariate in covariates:
        model_data = model_data.join(covariate, how='outer')
    
    return model_data.reset_index()


def create_pred_data(hierarchy: pd.DataFrame, adj_gbd_hierarchy: pd.DataFrame, population: pd.Series,
                     testing_capacity: pd.Series,
                     covariates: List[pd.Series],
                     pred_start_date: pd.Timestamp, pred_end_date: pd.Timestamp,
                     **kwargs):
    pred_data = pd.DataFrame(list(itertools.product(adj_gbd_hierarchy['location_id'].to_list(),
                                                    list(pd.date_range(pred_start_date, pred_end_date)))),
                         columns=['location_id', 'date'])
    pred_data['intercept'] = 1
    pred_data = pred_data.set_index(['location_id', 'date'])
    log_testing_rate_capacity = np.log(testing_capacity / population).rename('log_testing_rate_capacity')
    pred_data = pred_data.join(log_testing_rate_capacity, how='outer')
    
    for covariate in covariates:
        pred_data = pred_data.join(covariate, how='outer')
    
    return pred_data.reset_index()
