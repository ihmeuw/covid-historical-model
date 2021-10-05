from typing import Tuple, Dict, List

import pandas as pd
import numpy as np

from covid_historical_model.utils.math import logit, expit
from covid_historical_model.mrbrt import cascade
from covid_historical_model.rates import age_standardization
from covid_historical_model.etl import model_inputs
from covid_historical_model.rates.covariate_priors import get_covariate_priors, get_covariate_constraints


def run_model(model_data: pd.DataFrame, pred_data: pd.DataFrame,
              ifr_age_pattern: pd.Series, sero_age_pattern: pd.Series, age_spec_population: pd.Series,
              hierarchy: pd.DataFrame, gbd_hierarchy: pd.DataFrame,
              day_0: pd.Timestamp, day_inflection: pd.Timestamp,
              covariate_list: List[str],
              verbose: bool = True,
              **kwargs) -> Tuple[Dict, Dict, pd.Series, pd.Series, pd.Series]:
    model_data, age_stand_scaling_factor, level_lambdas, var_args, \
    global_prior_dict, pred_replace_dict, pred_exclude_vars = prepare_model(
        model_data=model_data,
        ifr_age_pattern=ifr_age_pattern,
        sero_age_pattern=sero_age_pattern,
        age_spec_population=age_spec_population,
        day_0=day_0,
        day_inflection=day_inflection,
        covariate_list=covariate_list,
    )
    mr_model_dict, prior_dicts = cascade.run_cascade(
        model_data=model_data.copy(),
        hierarchy=hierarchy.copy(),  # run w/ modeling hierarchy
        var_args=var_args.copy(),
        global_prior_dict=global_prior_dict.copy(),
        level_lambdas=level_lambdas.copy(),
        verbose=verbose,
    )
    adj_gbd_hierarchy = model_inputs.validate_hierarchies(hierarchy.copy(), gbd_hierarchy.copy())
    pred_data = pred_data.dropna()
    pred, pred_fe, pred_location_map = cascade.predict_cascade(
        pred_data=pred_data.copy(),
        hierarchy=adj_gbd_hierarchy.copy(),  # predict w/ gbd hierarchy
        mr_model_dict=mr_model_dict.copy(),
        pred_replace_dict=pred_replace_dict.copy(),
        pred_exclude_vars=pred_exclude_vars.copy(),
        var_args=var_args.copy(),
        verbose=verbose,
    )
    
    pred = expit(pred).rename(pred.name.replace('logit_', ''))
    pred_fe = expit(pred_fe).rename(pred_fe.name.replace('logit_', ''))
    
    pred /= age_stand_scaling_factor
    pred_fe /= age_stand_scaling_factor
    
    pred = pred.groupby(level=0).apply(lambda x: x * (1.0 / x.max()).clip(0, 1))
    pred_fe = pred_fe.groupby(level=0).apply(lambda x: x * (1.0 / x.max()).clip(0, 1))

    return mr_model_dict, prior_dicts, pred.dropna(), pred_fe.dropna(), pred_location_map, \
           age_stand_scaling_factor, level_lambdas


def prepare_model(model_data: pd.DataFrame,
                  ifr_age_pattern: pd.Series, sero_age_pattern: pd.Series, age_spec_population: pd.Series,
                  day_0: pd.Timestamp, day_inflection: pd.Timestamp,
                  covariate_list: List[str],):
    age_stand_scaling_factor = age_standardization.get_scaling_factor(
        ifr_age_pattern, sero_age_pattern,
        age_spec_population.loc[[1]], age_spec_population
    )
    model_data = model_data.set_index('location_id')
    model_data['ifr'] *= age_stand_scaling_factor[model_data.index]
    model_data['ifr'] *= model_data['ratio_data_scalar']
    model_data = model_data.reset_index()
    
    model_data['logit_ifr'] = logit(model_data['ifr'])
    model_data['logit_ifr'] = model_data['logit_ifr'].replace((-np.inf, np.inf), np.nan)
    model_data['ifr_se'] = 1
    model_data['logit_ifr_se'] = 1
    model_data['intercept'] = 1
    
    # lose 0s and 1s
    model_data = model_data.loc[model_data['logit_ifr'].notnull()]

    inflection_point = (day_inflection - day_0).days
    inflection_point = (inflection_point - model_data['t'].min()) / model_data['t'].values.ptp()
    
    covariate_priors = get_covariate_priors(1, 'ifr')
    covariate_priors = {covariate: covariate_priors[covariate] for covariate in covariate_list}
    covariate_constraints = get_covariate_constraints('ifr')
    covariate_constraints = {covariate: covariate_constraints[covariate] for covariate in covariate_list}
    covariate_lambdas = {covariate: 1. for covariate in covariate_list}

    var_args = {'dep_var': 'logit_ifr',
                'dep_var_se': 'logit_ifr_se',
                'fe_vars': ['intercept', 't',] + covariate_list,  # 'obesity'
                'prior_dict': {'t': {'use_spline': True,
                                     'spline_knots_type': 'domain',
                                     'spline_knots': np.array([0., inflection_point, 1.]),
                                     'spline_degree': 1,
                                     'prior_spline_maxder_uniform': np.array([[-np.inf, -0.],
                                                                              [-1e-6  , 0. ]]),
                                    },
                               **covariate_constraints,
                              },
                're_vars': [],
                'group_var': 'location_id',}
    global_prior_dict = {'t': {'prior_spline_maxder_gaussian': np.array([[-2e-3, 0.    ],
                                                                         [ 1e-3, np.inf]]),},
                         **covariate_priors,}
    pred_replace_dict = {}
    pred_exclude_vars = []
    level_lambdas = {
        # fit covariates at global level, tight lambdas after
        0: {'intercept': 2.  , 't': 0.5, **covariate_lambdas,},  # G->SR
        1: {'intercept': 2.  , 't': 1. , **covariate_lambdas,},  # SR->R
        2: {'intercept': 100., 't': 10., **covariate_lambdas,},  # R->A0
        3: {'intercept': 100., 't': 10., **covariate_lambdas,},  # A0->A1
        4: {'intercept': 100., 't': 10., **covariate_lambdas,},  # A1->A2
        5: {'intercept': 100., 't': 10., **covariate_lambdas,},  # A2->A3
    }
    
    
    if var_args['group_var'] != 'location_id':
        raise ValueError('NRMSE data assignment assumes `study_id` == `location_id` (`location_id` must be group_var).')
    
    # SUPPRESSING CASCADE CONSOLE OUTPUT
    model_data_cols = ['location_id', 'date', var_args['dep_var'],
                       var_args['dep_var_se']] + var_args['fe_vars']
    model_data = model_data.loc[:, model_data_cols]
    model_data = model_data.dropna()
    
    return model_data, age_stand_scaling_factor, level_lambdas, var_args, global_prior_dict,\
           pred_replace_dict, pred_exclude_vars
