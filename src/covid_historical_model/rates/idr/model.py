from typing import Tuple, Dict

import pandas as pd
import numpy as np

from covid_historical_model.utils.math import logit, expit
from covid_historical_model.mrbrt import cascade


def run_model(model_data: pd.DataFrame, pred_data: pd.DataFrame,
              hierarchy: pd.DataFrame,
              verbose: bool = True,
              **kwargs) -> Tuple[Dict, Dict, pd.Series, pd.Series, pd.Series]:
    model_data['logit_idr'] = logit(model_data['idr'])
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
        0: {'intercept': 10., 'log_infwavg_testing_rate_capacity': 100.,},
        1: {'intercept': 10., 'log_infwavg_testing_rate_capacity': 100.,},
        2: {'intercept': 10., 'log_infwavg_testing_rate_capacity': 100.,},
        3: {'intercept': 10., 'log_infwavg_testing_rate_capacity': 100.,},
        4: {'intercept': 10., 'log_infwavg_testing_rate_capacity': 100.,},
    }
    
    if var_args['group_var'] != 'location_id':
        raise ValueError('NRMSE data assignment assumes `study_id` == `location_id` (`location_id` must be group_var).')
        
    # SUPPRESSING CASCADE CONSOLE OUTPUT
    model_data_cols = ['location_id', 'date', var_args['dep_var'],
                       var_args['dep_var_se']] + var_args['fe_vars']
    model_data = model_data.loc[:, model_data_cols]
    model_data = model_data.dropna()
    mr_model_dicts, prior_dicts = cascade.run_cascade(
        model_data=model_data.copy(),
        hierarchy=hierarchy.copy(),
        var_args=var_args.copy(),
        level_lambdas=level_lambdas.copy(),
        verbose=False,
    )
    pred_data = pred_data.dropna()
    pred, pred_fe, pred_location_map = cascade.predict_cascade(
        pred_data=pred_data.copy(),
        hierarchy=hierarchy.copy(),
        mr_model_dicts=mr_model_dicts.copy(),
        pred_replace_dict=pred_replace_dict.copy(),
        pred_exclude_vars=pred_exclude_vars.copy(),
        var_args=var_args.copy(),
        verbose=False,
    )
    
    pred = expit(pred).rename(pred.name.replace('logit_', ''))
    pred_fe = expit(pred_fe).rename(pred.name.replace('logit_', ''))

    return mr_model_dicts, prior_dicts, pred.dropna(), pred_fe.dropna(), pred_location_map
