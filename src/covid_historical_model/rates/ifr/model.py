from typing import Tuple, Dict, List

import pandas as pd
import numpy as np

from covid_historical_model.utils.math import logit, expit
from covid_historical_model.mrbrt import cascade
from covid_historical_model.rates import age_standardization
from covid_historical_model.durations.durations import SERO_TO_DEATH


def run_model(model_data: pd.DataFrame, pred_data: pd.DataFrame,
              ifr_age_pattern: pd.Series, sero_age_pattern: pd.Series, age_spec_population: pd.Series,
              hierarchy: pd.DataFrame,
              day_0: pd.Timestamp, day_inflection: pd.Timestamp,
              verbose: bool = True,
              **kwargs) -> Tuple[Dict, Dict, pd.Series, pd.Series, pd.Series]:
    age_stand_scaling_factor = age_standardization.get_scaling_factor(
        ifr_age_pattern, sero_age_pattern,
        age_spec_population.loc[[1]], age_spec_population
    )
    model_data = model_data.set_index('location_id')
    model_data['ifr'] *= age_stand_scaling_factor[model_data.index]
    model_data = model_data.reset_index()
    
    model_data['logit_ifr'] = logit(model_data['ifr'])
    model_data['logit_ifr'] = model_data['logit_ifr'].replace((-np.inf, np.inf), np.nan)
    model_data['ifr_se'] = 1
    model_data['logit_ifr_se'] = 1
    model_data['intercept'] = 1

    inflection_point = (day_inflection - day_0).days
    inflection_point /= model_data['t'].values.ptp()

    var_args = {'dep_var': 'logit_ifr',
                'dep_var_se': 'logit_ifr_se',
                'fe_vars': ['intercept', 't','obesity'],
                'prior_dict': {'t':{'use_spline': True,
                                    'spline_knots_type': 'domain',
                                    'spline_knots': np.array([0., inflection_point, 1.]),
                                    'spline_degree': 1,
                                    'prior_spline_maxder_uniform': np.array([[-np.inf, -0.],
                                                                             [-1e-6  , -0.]]),
                                    'prior_spline_maxder_gaussian': np.array([[-2e-3, 0.    ],
                                                                              [0.01 , np.inf]]),
                                   },
                               'obesity':{'prior_beta_uniform': np.array([np.log(1.3), np.log(1.3)])},
                              },
                're_vars': [],
                'group_var': 'location_id',}
    pred_replace_dict = {}
    pred_exclude_vars = []
    level_lambdas = {
        0: {'intercept': 1.  , 't': 1., 'obesity': 1.,},
        1: {'intercept': 1.  , 't': 1., 'obesity': 1.,},
        2: {'intercept': 100., 't': 5., 'obesity': 1.,},
        3: {'intercept': 100., 't': 5., 'obesity': 1.,},
        4: {'intercept': 100., 't': 5., 'obesity': 1.,},
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
        hierarchy=hierarchy.copy(),
        mr_model_dict=mr_model_dict.copy(),
        pred_replace_dict=pred_replace_dict.copy(),
        pred_exclude_vars=pred_exclude_vars.copy(),
        var_args=var_args.copy(),
        verbose=False,
    )
    
    pred = expit(pred).rename(pred.name.replace('logit_', ''))
    pred_fe = expit(pred_fe).rename(pred.name.replace('logit_', ''))
    
    pred /= age_stand_scaling_factor
    pred_fe /= age_stand_scaling_factor

    return mr_model_dict, prior_dicts, pred.dropna(), pred_fe.dropna(), pred_location_map


def match_model_data(mr_model_dict: Dict, data: pd.Series,
                     pred_location_map: Dict, model_location_id: int,
                     date_req: pd.Timestamp = pd.Timestamp('2020-07-01'),
                     obs_req: int = 2) -> Tuple[List, bool]:
    '''
    Must have `obs_req` observations after `date_req`.
    '''
    model_data_location_ids = mr_model_dict[model_location_id].data.to_df()['study_id'].unique().tolist()
    data = data.loc[model_data_location_ids].reset_index()
    
    satisfies = (data['date'] >= date_req).sum() >= obs_req
    
    return model_data_location_ids, satisfies


def map_pred_and_model_locations(pred_location_map: Dict, mr_model_dict: Dict,
                                 data: pd.Series, hierarchy: pd.DataFrame) -> pd.DataFrame:
    nrmse_data = []
    for pred_location_id in pred_location_map.keys():
        model_location_id = pred_location_id
        searching = True
        while searching:
            model_location_id = pred_location_map[model_location_id]
            model_data_location_ids, satisfies = match_model_data(
                mr_model_dict, data.copy(), pred_location_map, model_location_id,
            )
            searching = not satisfies
            if searching:
                model_location_id = hierarchy.loc[hierarchy['location_id'] == model_location_id, 'parent_id'].item()
        nrmse_data.append(
            pd.concat([data.loc[model_data_location_ids].reset_index(),
                       pd.DataFrame({'pred_location_id':pred_location_id},
                                    index=np.arange(len(data.loc[model_data_location_ids])))],
                      axis=1)
        )
    nrmse_data = pd.concat(nrmse_data).reset_index(drop=True)
    nrmse_data = (nrmse_data
                  .set_index(['pred_location_id', 'location_id', 'date'])
                  .sort_index())
    
    return nrmse_data


def get_nrmse(seroprevalence: pd.DataFrame, deaths: pd.Series,
              pred: pd.Series, hierarchy: pd.DataFrame, population: pd.Series,
              pred_location_map: pd.Series, mr_model_dict: Dict) -> pd.Series:
    seroprevalence = (seroprevalence
                      .set_index(['location_id', 'date'])
                      .sort_index()
                      .loc[:, 'seroprevalence'])
    
    infections = (deaths / pred).dropna().rename('infections')
    infections = infections.reset_index()
    infections['date'] -= pd.Timedelta(days=SERO_TO_DEATH)  # matching date to serosurveys
    infections = infections.set_index(['location_id', 'date'])
    infections = infections.groupby(level=0)['infections'].cumsum()
    infections /= population
    
    residuals = seroprevalence.to_frame().join(infections)
    residuals = (residuals['seroprevalence'] - residuals['infections']).dropna().rename('residuals')
    residuals = map_pred_and_model_locations(pred_location_map, mr_model_dict, residuals, hierarchy)
    residuals = residuals.loc[:, 'residuals']
    seroprevalence = map_pred_and_model_locations(pred_location_map, mr_model_dict, seroprevalence, hierarchy)
    seroprevalence = seroprevalence.loc[:, 'seroprevalence']
    seroprevalence = seroprevalence[residuals.index]
    rmse = np.sqrt((residuals ** 2).groupby(level=0).mean())
    
    s_min = seroprevalence.groupby(level=0).min()
    s_max = seroprevalence.groupby(level=0).max()
    s_ptp = s_max - s_min
    s_ptp = s_ptp.replace(0, np.nan)
    s_ptp = s_ptp.fillna((s_max + s_min) / 2)
    
    nrmse = (rmse / s_ptp).dropna().rename('nrmse')
    
    nrmse.index = nrmse.index.rename('location_id')
    
    return nrmse, residuals
