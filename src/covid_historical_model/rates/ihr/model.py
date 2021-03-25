from typing import Tuple, Dict

import pandas as pd
import numpy as np

from covid_historical_model.utils.math import logit, expit
from covid_historical_model.mrbrt import cascade
from covid_historical_model.rates import age_standardization


def run_model(model_data: pd.DataFrame, pred_data: pd.DataFrame,
              ihr_age_pattern: pd.Series, sero_age_pattern: pd.Series, age_spec_population: pd.Series,
              hierarchy: pd.DataFrame,
              verbose: bool = True,
              **kwargs) -> Tuple[Dict, Dict, pd.Series, pd.Series, pd.Series]:
    age_stand_scaling_factor = age_standardization.get_scaling_factor(
        ihr_age_pattern, sero_age_pattern,
        age_spec_population.loc[[1]], age_spec_population
    )
    model_data = model_data.set_index('location_id')
    model_data['ihr'] *= age_stand_scaling_factor[model_data.index]
    model_data = model_data.reset_index()
    
    model_data['logit_ihr'] = logit(model_data['ihr'])
    model_data['ihr_se'] = 1
    model_data['logit_ihr_se'] = 1
    model_data['intercept'] = 1

    var_args = {'dep_var': 'logit_ihr',
                'dep_var_se': 'logit_ihr_se',
                'fe_vars': ['intercept'],
                'prior_dict': {},
                're_vars': [],
                'group_var': 'location_id',
                }
    pred_replace_dict = {}
    pred_exclude_vars = []
    level_lambdas = {
        0: {'intercept': 100.,},
        1: {'intercept': 100.,},
        2: {'intercept': 100.,},
        3: {'intercept': 100.,},
        4: {'intercept': 100.,},
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
    
    pred /= age_stand_scaling_factor
    pred_fe /= age_stand_scaling_factor

    return mr_model_dicts, prior_dicts, pred.dropna(), pred_fe.dropna(), pred_location_map
