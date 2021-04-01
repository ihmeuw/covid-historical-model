from pathlib import Path
from loguru import logger

import pandas as pd
import numpy as np

from covid_historical_model.etl import helpers


def testing(testing_root: Path) -> pd.DataFrame:
    data_path = testing_root / 'forecast_raked_test_pc_simple.csv'
    data = pd.read_csv(data_path)
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values(['location_id', 'date']).reset_index(drop=True)
    data['daily_tests'] = data['test_pc'] * data['pop']
    data['cumulative_tests'] = data.groupby('location_id')['daily_tests'].cumsum()
    data = (data
            .loc[:, ['location_id', 'date', 'cumulative_tests']]
            .sort_values(['location_id', 'date'])
            .reset_index(drop=True))
    data = (data.groupby('location_id', as_index=False)
            .apply(lambda x: helpers.fill_dates(x, ['cumulative_tests']))
            .reset_index(drop=True))
    data = data.sort_values(['location_id', 'date']).reset_index(drop=True)
    data['daily_tests'] = (data
                           .groupby('location_id')['cumulative_tests']
                           .apply(lambda x: x.diff()))
    data = data.dropna()
    data = data.sort_values(['location_id', 'date']).reset_index(drop=True)
    data['testing_capacity'] = data.groupby('location_id')['daily_tests'].cummax()

    data = (data
            .set_index(['location_id', 'date'])
            .sort_index()
            .loc[:, ['daily_tests', 'testing_capacity', 'cumulative_tests']])
    
    return data


def ihr_age_pattern(age_pattern_root: Path) -> pd.Series:
    data_path = age_pattern_root / 'hir_preds_5yr.csv'
    data = pd.read_csv(data_path)
    
    data = data.rename(columns={'age_group_start': 'age_group_years_start',
                                'age_group_end': 'age_group_years_end',
                                'hir': 'ihr',})
    data['age_group_years_end'].iloc[-1] = 125

    data = (data
            .set_index(['age_group_years_start', 'age_group_years_end'])
            .sort_index()
            .loc[:, 'ihr'])
    
    return data


def ifr_age_pattern(age_pattern_root: Path) -> pd.Series:
    data_path = age_pattern_root / 'ifr_preds_5yr.csv'
    data = pd.read_csv(data_path)
    
    data = data.rename(columns={'age_group_start': 'age_group_years_start',
                                'age_group_end': 'age_group_years_end',})
    data['age_group_years_end'].iloc[-1] = 125

    data = (data
            .set_index(['age_group_years_start', 'age_group_years_end'])
            .sort_index()
            .loc[:, 'ifr'])
    
    return data


def seroprevalence_age_pattern(age_pattern_root: Path) -> pd.Series:
    data_path = age_pattern_root / 'seroprev_preds_5yr.csv'
    data = pd.read_csv(data_path)
    
    data = data.rename(columns={'age_group_start': 'age_group_years_start',
                                'age_group_end': 'age_group_years_end',
                                'seroprev': 'seroprevalence',})
    data['age_group_years_end'].iloc[-1] = 125

    data = (data
            .set_index(['age_group_years_start', 'age_group_years_end'])
            .sort_index()
            .loc[:, 'seroprevalence'])
    
    return data


def vaccine_coverage(vaccine_coverage_root: Path) -> pd.DataFrame:
    data_path = vaccine_coverage_root / 'slow_scenario_vaccine_coverage.csv'
    data = pd.read_csv(data_path)
    data['date'] = pd.to_datetime(data['date'])
    
    keep_columns = [
        # total seroconverted
        'cumulative_all_effective',
        
        # elderly (mutually exclusive)
        'cumulative_hr_effective_wildtype',
        'cumulative_hr_effective_protected_wildtype',
        'cumulative_hr_effective_variant',
        'cumulative_hr_effective_protected_variant',
    
        # other adults (mutually exclusive)
        'cumulative_lr_effective_wildtype',
        'cumulative_lr_effective_protected_wildtype',
        'cumulative_lr_effective_variant',
        'cumulative_lr_effective_protected_variant',
    ]
    
    data = (data
            .set_index(['location_id', 'date'])
            .sort_index()
            .loc[:, keep_columns])
    
    return data


def variant_scaleup(variant_scaleup_root: Path, variant_type: str, verbose: bool = True) -> pd.Series:
    data_path = variant_scaleup_root / 'variant_reference.csv'
    data = pd.read_csv(data_path)
    data['date'] = pd.to_datetime(data['date'])
    
    if variant_type == 'escape':
        is_escape_variant = ~data['variant'].isin(['wild_type', 'B117'])
        data = data.loc[is_escape_variant]
        if verbose:
            logger.info(f"Escape variants: {', '.join(data['variant'].unique())}")
        data = data.rename(columns={'prevalence': 'escape_variant_prevalence'})
        data = data.groupby(['location_id', 'date'])['escape_variant_prevalence'].sum()
    elif variant_type == 'severity':
        is_variant = data['variant'].isin(['B117'])
        data = data.loc[is_variant]
        if verbose:
            logger.info(f"Variants: {', '.join(data['variant'].unique())}")
        data = data.rename(columns={'prevalence': 'severity_variant_prevalence'})
        data = data.groupby(['location_id', 'date'])['severity_variant_prevalence'].sum()
    else:
        raise ValueError(f'Invalid variant type specified: {variant_type}')

    
    return data
