from typing import Callable
import pandas as pd


def get_all_age_rate(rate_age_pattern: pd.Series, infection_age_pattern: pd.Series,
                     age_spec_population: pd.Series,) -> pd.Series:
    infections_by_age = age_spec_population.to_frame().join(infection_age_pattern)
    infections_by_age = infections_by_age['population'] * infections_by_age['seroprevalence']
    infections_by_age = infections_by_age.rename('infections')
    
    infections_age_weights = infections_by_age / infections_by_age.groupby(level=2).sum()
    infections_age_weights = infections_age_weights.rename('age_weight')
    
    all_age_rate = infections_age_weights.to_frame().join(rate_age_pattern.rename('rate'))
    all_age_rate = (all_age_rate['rate'] * all_age_rate['age_weight']).groupby(level=2).sum()
    all_age_rate = all_age_rate.rename('rate')
    
    return all_age_rate


def get_scaling_factor(rate_age_pattern: pd.Series, infection_age_pattern: pd.Series,
                       global_age_spec_population: pd.Series,
                       local_age_spec_population: pd.Series,) -> pd.Series:
    if rate_age_pattern.index.names != ['age_group_years_start', 'age_group_years_end']:
        raise ValueError('Wrong rate_age_pattern indexes in age-stadardization.')
    if infection_age_pattern.index.names != ['age_group_years_start', 'age_group_years_end']:
        raise ValueError('Wrong infection_age_pattern indexes in age-stadardization.')
    if global_age_spec_population.index.names != ['location_id', 'age_group_years_start', 'age_group_years_end']:
        raise ValueError('Wrong global_age_spec_population indexes in age-stadardization.')
    if local_age_spec_population.index.names != ['location_id', 'age_group_years_start', 'age_group_years_end']:
        raise ValueError('Wrong local_age_spec_population indexes in age-stadardization.')
    
    global_all_age_rate = get_all_age_rate(rate_age_pattern, infection_age_pattern, global_age_spec_population)
    local_all_age_rate = get_all_age_rate(rate_age_pattern, infection_age_pattern, local_age_spec_population)
    
    scaling_factor = global_all_age_rate.loc[1] / local_all_age_rate
    
    return scaling_factor.rename('scaling_factor')


def split_age_specific_data(data: pd.Series, age_start_filter: Callable) -> pd.Series:
    data_col = data.name
    idx_cols = data.index.names
    data = data.reset_index()
    to_include = data['age_group_years_start'].apply(age_start_filter)
    
    data = (data
            .loc[to_include]
            .set_index(idx_cols)
            .sort_index()
            .loc[:, data_col])
    
    return data


def get_risk_group_rr(rate_age_pattern: pd.Series, infection_age_pattern: pd.Series,
                      age_spec_population: pd.Series,
                      high_risk_age_group_years_start: int = 65,):
    all_age_rate = get_all_age_rate(rate_age_pattern, infection_age_pattern, age_spec_population)
    
    low_risk_rate = get_all_age_rate(
        split_age_specific_data(rate_age_pattern.copy(), lambda x: x < high_risk_age_group_years_start),
        split_age_specific_data(infection_age_pattern.copy(), lambda x: x < high_risk_age_group_years_start),
        split_age_specific_data(age_spec_population.copy(), lambda x: x < high_risk_age_group_years_start)
    )
    
    high_risk_rate = get_all_age_rate(
        split_age_specific_data(rate_age_pattern.copy(), lambda x: x >= high_risk_age_group_years_start),
        split_age_specific_data(infection_age_pattern.copy(), lambda x: x >= high_risk_age_group_years_start),
        split_age_specific_data(age_spec_population.copy(), lambda x: x >= high_risk_age_group_years_start)
    )
    
    low_risk_rr = (low_risk_rate / all_age_rate).rename('low_risk_rr')
    high_risk_rr = (high_risk_rate / all_age_rate).rename('high_risk_rr')
    
    return low_risk_rr, high_risk_rr
