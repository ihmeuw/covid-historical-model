from pathlib import Path
from typing import Dict, List
import itertools

import pandas as pd
import numpy as np

from covid_historical_model.etl import model_inputs, estimates
from covid_historical_model.durations.durations import ADMISSION_TO_SERO


def load_input_data(model_inputs_root: Path, age_pattern_root: Path,
                    seroprevalence: pd.DataFrame, vaccine_coverage: pd.DataFrame,
                    escape_variant_prevalence: pd.Series, severity_variant_prevalence: pd.Series,
                    verbose: bool = True) -> Dict:
    # load data
    hierarchy = model_inputs.hierarchy(model_inputs_root)
    gbd_hierarchy = model_inputs.hierarchy(model_inputs_root, 'covid_gbd')
    population = model_inputs.population(model_inputs_root)
    age_spec_population = model_inputs.population(model_inputs_root, by_age=True)
    cumulative_hospitalizations, daily_hospitalizations = model_inputs.reported_epi(
        model_inputs_root, 'hospitalizations', hierarchy
    )
    sero_age_pattern = estimates.seroprevalence_age_pattern(age_pattern_root)
    ihr_age_pattern = estimates.ihr_age_pattern(age_pattern_root)
    
    covariates = []
    
    return {'cumulative_hospitalizations': cumulative_hospitalizations,
            'daily_hospitalizations': daily_hospitalizations,
            'seroprevalence': seroprevalence,
            'vaccine_coverage': vaccine_coverage,
            'covariates': covariates,
            'sero_age_pattern': sero_age_pattern,
            'ihr_age_pattern': ihr_age_pattern,
            'age_spec_population': age_spec_population,
            'escape_variant_prevalence': escape_variant_prevalence,
            'severity_variant_prevalence': severity_variant_prevalence,
            'hierarchy': hierarchy,
            'gbd_hierarchy': gbd_hierarchy,
            'population': population,}


def create_model_data(cumulative_hospitalizations: pd.Series,
                      daily_hospitalizations: pd.Series,
                      seroprevalence: pd.DataFrame,
                      covariates: List[pd.Series],
                      hierarchy: pd.DataFrame, population: pd.Series,
                      day_0: pd.Timestamp,
                      **kwargs) -> pd.DataFrame:
    ihr_data = seroprevalence.loc[seroprevalence['is_outlier'] == 0].copy()
    ihr_data['date'] -= pd.Timedelta(days=ADMISSION_TO_SERO)
    ihr_data = (ihr_data
                .set_index(['location_id', 'date'])
                .loc[:, 'seroprevalence'])
    ihr_data = ((cumulative_hospitalizations / (ihr_data * population))
                .dropna()
                .rename('ihr'))

    # get mean day of admission int
    loc_dates = ihr_data.index.drop_duplicates().to_list()
    time = []
    for location_id, survey_end_date in loc_dates:
        lochosps = daily_hospitalizations.loc[location_id]
        lochosps = lochosps.clip(0, np.inf)
        lochosps = lochosps.reset_index()
        lochosps = lochosps.loc[lochosps['date'] <= survey_end_date]
        lochosps['t'] = (lochosps['date'] - day_0).dt.days
        t = np.average(lochosps['t'], weights=lochosps['daily_hospitalizations'] + 1e-6)
        t = int(np.round(t))
        mean_hospitalization_date = lochosps.loc[lochosps['t'] == t, 'date'].item()
        time.append(
            pd.DataFrame(
                {'t':t, 'mean_hospitalization_date':mean_hospitalization_date},
                index=pd.MultiIndex.from_arrays([[location_id], [survey_end_date]],
                                                names=('location_id', 'date')),)
        )
    time = pd.concat(time)

    # add time
    model_data = time.join(ihr_data, how='outer')
    
    # add covariates
    for covariate in covariates:
        model_data = model_data.join(covariate, how='outer')
            
    return model_data.reset_index()


def create_pred_data(hierarchy: pd.DataFrame, gbd_hierarchy: pd.DataFrame, population: pd.Series,
                     covariates: List[pd.Series],
                     pred_start_date: pd.Timestamp, pred_end_date: pd.Timestamp,
                     day_0: pd.Timestamp,
                     **kwargs):
    adj_gbd_hierarchy = model_inputs.validate_hierarchies(hierarchy.copy(), gbd_hierarchy.copy())
    pred_data = pd.DataFrame(list(itertools.product(adj_gbd_hierarchy['location_id'].to_list(),
                                                    list(pd.date_range(pred_start_date, pred_end_date)))),
                         columns=['location_id', 'date'])
    pred_data['intercept'] = 1
    pred_data['t'] = (pred_data['date'] - day_0).dt.days
    pred_data = pred_data.set_index(['location_id', 'date'])
    
    for covariate in covariates:
        pred_data = pred_data.join(covariate, how='outer')
    
    return pred_data.reset_index()
