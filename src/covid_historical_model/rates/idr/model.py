from typing import Tuple, Dict, List

import pandas as pd
import numpy as np

from covid_historical_model.utils.math import logit, expit
from covid_historical_model.mrbrt import cascade


def run_model(model_data: pd.DataFrame, pred_data: pd.DataFrame,
              hierarchy: pd.DataFrame, cov_hierarchy: pd.DataFrame,
              verbose: bool = True,
              **kwargs) -> Tuple[Dict, Dict, pd.Series, pd.Series, pd.Series]:
    model_data['logit_idr'] = logit(model_data['idr'])
    model_data['logit_idr'] = model_data['logit_idr'].replace((-np.inf, np.inf), np.nan)
    model_data['idr_se'] = 1
    model_data['logit_idr_se'] = 1
    model_data['intercept'] = 1

    var_args = {'dep_var': 'logit_idr',
                'dep_var_se': 'logit_idr_se',
                'fe_vars': ['intercept',  # , 'bias'
                            'log_infwavg_testing_rate_capacity',],
                'prior_dict': {'log_infwavg_testing_rate_capacity':
                                   {'prior_beta_uniform':np.array([1e-6, np.inf])},
                              },
                're_vars': [],
                'group_var': 'location_id',}
    pred_replace_dict = {'log_testing_rate_capacity': 'log_infwavg_testing_rate_capacity',}
    pred_exclude_vars = []
    level_lambdas = {
        0: {'intercept': 5., 'log_infwavg_testing_rate_capacity': 100.,},
        1: {'intercept': 5., 'log_infwavg_testing_rate_capacity': 100.,},
        2: {'intercept': 5., 'log_infwavg_testing_rate_capacity': 100.,},
        3: {'intercept': 5., 'log_infwavg_testing_rate_capacity': 100.,},
        4: {'intercept': 5., 'log_infwavg_testing_rate_capacity': 100.,},
    }
    
    if var_args['group_var'] != 'location_id':
        raise ValueError('NRMSE data assignment assumes `study_id` == `location_id` (`location_id` must be group_var).')
        
    # SUPPRESSING CASCADE CONSOLE OUTPUT
    model_data_cols = ['location_id', 'date', var_args['dep_var'],
                       var_args['dep_var_se']] + var_args['fe_vars']
    model_data = model_data.loc[:, model_data_cols]
    model_data = model_data.dropna()
    mr_model_dict, prior_dicts = cascade.run_cascade(
        model_data=model_data.copy(),
        hierarchy=hierarchy.copy(),
        var_args=var_args.copy(),
        level_lambdas=level_lambdas.copy(),
        verbose=False,
    )
    pred_data = pred_data.dropna()
    pred, pred_fe, pred_location_map = cascade.predict_cascade(
        pred_data=pred_data.copy(),
        hierarchy=cov_hierarchy.copy(),
        mr_model_dict=mr_model_dict.copy(),
        pred_replace_dict=pred_replace_dict.copy(),
        pred_exclude_vars=pred_exclude_vars.copy(),
        var_args=var_args.copy(),
        verbose=False,
    )
    
    pred = expit(pred).rename(pred.name.replace('logit_', ''))
    pred_fe = expit(pred_fe).rename(pred.name.replace('logit_', ''))

    return mr_model_dict, prior_dicts, pred.dropna(), pred_fe.dropna(), pred_location_map


def determine_mean_date_of_infection(location_dates: List,
                                     daily_cases: pd.DataFrame,
                                     pred: pd.Series) -> pd.DataFrame:
    daily_infections = (daily_cases / pred).rename('daily_infections').dropna()

    dates_data = []
    for location_id, date in location_dates:
        data = daily_infections[location_id]
        data = data.reset_index()
        data = data.loc[data['date'] <= date].reset_index(drop=True)
        if not data.empty:
            avg_date_of_infection_idx = int(np.round(np.average(data.index, weights=(data['daily_infections'] + 1))))
            avg_date_of_infection = data.loc[avg_date_of_infection_idx, 'date']
            dates_data.append(pd.DataFrame({'location_id':location_id, 'date':date, 'avg_date_of_infection':avg_date_of_infection}, index=[0]))
    dates_data = pd.concat(dates_data).reset_index(drop=True)

    return dates_data
