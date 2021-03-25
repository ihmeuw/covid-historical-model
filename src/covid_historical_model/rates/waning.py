import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from covid_historical_model.durations.durations import SERO_TO_DEATH

## TODO:
##     - add diagnostics


def fit_sensitivity_decay(t: np.array, sensitivity: np.array, increasing: bool, t_N: int = 720) -> pd.DataFrame:
    def sigmoid(x, x0, k):
        y = 1 / (1 + np.exp(-k * (x-x0)))
        return y
    
    if increasing:
        bounds = ([-np.inf, 1e-4], [np.inf, 0.5])
    else:
        bounds = ([-np.inf, -0.5], [np.inf, -1e-4])
    popt, pcov = curve_fit(sigmoid,
                           t, sensitivity,
                           method='dogbox',
                           bounds=bounds, max_nfev=1000)
    
    t_pred = np.arange(0, t_N + 1)
    sensitivity_pred = sigmoid(t_pred, *popt)
        
    return pd.DataFrame({'t': t_pred, 'sensitivity': sensitivity_pred})


def fit_hospital_weighted_sensitivity_decay(sensitivity: pd.DataFrame, increasing: bool, hospitalized_weights: pd.Series) -> pd.DataFrame:
    assay = sensitivity['assay'].unique().item()
    hosp_sensitivity = sensitivity.loc[sensitivity['hospitalization_status'] == 'Hospitalized']
    nonhosp_sensitivity = sensitivity.loc[sensitivity['hospitalization_status'] == 'Non-hospitalized']
    hosp_sensitivity = fit_sensitivity_decay(hosp_sensitivity['t'].values,
                                             hosp_sensitivity['sensitivity'].values,
                                             increasing)
    nonhosp_sensitivity = fit_sensitivity_decay(nonhosp_sensitivity['t'].values,
                                                nonhosp_sensitivity['sensitivity'].values,
                                                increasing)
    sensitivity = (hosp_sensitivity
                   .rename(columns={'sensitivity':'hosp_sensitivity'})
                   .merge(nonhosp_sensitivity
                          .rename(columns={'sensitivity':'nonhosp_sensitivity'})))
    sensitivity['key'] = 0
    hospitalized_weights = hospitalized_weights.rename('hospitalized_weights')
    hospitalized_weights = hospitalized_weights.reset_index()
    hospitalized_weights['key'] = 0
    sensitivity = sensitivity.merge(hospitalized_weights, on='key', how='outer')
    sensitivity['sensitivity'] = (sensitivity['hosp_sensitivity'] * sensitivity['hospitalized_weights']) + \
                                 (sensitivity['nonhosp_sensitivity'] * (1 - sensitivity['hospitalized_weights']))
    sensitivity = sensitivity.reset_index()
    sensitivity['assay'] = assay
    
    return sensitivity.loc[:, ['location_id', 'assay', 't', 'sensitivity', 'hosp_sensitivity', 'nonhosp_sensitivity']]


def calulate_waning_factor(infections: pd.DataFrame, sensitivity: pd.DataFrame,
                           sero_date: pd.Timestamp) -> float:
    infections['t'] = (sero_date - infections['date']).dt.days
    infections = infections.merge(sensitivity.reset_index())
    waning_factor = infections['infections'].sum() / (infections['infections'] * infections['sensitivity']).sum()
    waning_factor = max(1, waning_factor)

    return waning_factor
    
    
def adjust_location_seroprevalence(infections: pd.DataFrame, sensitivity: pd.DataFrame,
                                   seroprevalence: pd.DataFrame) -> pd.DataFrame:
    adj_seroprevalence = []
    for i, (sero_data_id, sero_date, sero_value) in enumerate(zip(seroprevalence['data_id'], seroprevalence['date'], seroprevalence['seroprevalence'])):
        waning_factor = calulate_waning_factor(infections, sensitivity, sero_date)
        adj_seroprevalence.append(pd.DataFrame({
            'data_id': sero_data_id,
            'date': sero_date,
            'seroprevalence': sero_value * waning_factor
        }, index=[i]))
    adj_seroprevalence = pd.concat(adj_seroprevalence)
    
    return adj_seroprevalence


def adjust_seroprevalence(ifr: pd.Series, daily_deaths: pd.Series, sensitivity: pd.DataFrame,
                          seroprevalence: pd.DataFrame) -> pd.DataFrame:
    infections = ((daily_deaths / ifr)
                  .dropna()
                  .rename('infections')
                  .reset_index()
                  .set_index('location_id'))
    infections['date'] -= pd.Timedelta(days=SERO_TO_DEATH)
    
    seroprevalence_list = []
    location_ids = seroprevalence['location_id'].unique().tolist()
    location_ids = [location_id for location_id in location_ids if location_id in infections.reset_index()['location_id'].to_list()]
    for location_id in location_ids:
        _sero = adjust_location_seroprevalence(infections.loc[location_id],
                                               sensitivity.loc[location_id],
                                               (seroprevalence
                                                .loc[seroprevalence['location_id'] == location_id, ['data_id', 'date', 'seroprevalence']]
                                                .reset_index(drop=True)))
        _sero['location_id'] = location_id
        seroprevalence_list.append(_sero)
    seroprevalence = pd.concat(seroprevalence_list).reset_index(drop=True)
    
    return seroprevalence
