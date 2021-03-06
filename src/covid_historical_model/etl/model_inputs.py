from pathlib import Path
from typing import Dict, Tuple
from loguru import logger

import pandas as pd
import numpy as np

from covid_historical_model.etl import db, helpers


def evil_doings(data: pd.DataFrame, hierarchy: pd.DataFrame, input_measure: str) -> Tuple[pd.DataFrame, Dict]:
    manipulation_metadata = {}
    if input_measure == 'cases':
        pass

    elif input_measure == 'hospitalizations':
        # ## hosp/IHR == admissions too low
        # is_argentina = data['location_id'] == 97
        # data = data.loc[~is_argentina].reset_index(drop=True)
        # manipulation_metadata['argentina'] = 'dropped all hospitalizations'
                
        ## is just march-june 2020
        is_vietnam = data['location_id'] == 20
        data = data.loc[~is_vietnam].reset_index(drop=True)
        manipulation_metadata['vietnam'] = 'dropped all hospitalizations'

        ## is just march-june 2020
        is_murcia = data['location_id'] == 60366
        data = data.loc[~is_murcia].reset_index(drop=True)
        manipulation_metadata['murcia'] = 'dropped all hospitalizations'

        ## partial time series
        pakistan_location_ids = hierarchy.loc[hierarchy['path_to_top_parent'].apply(lambda x: '165' in x.split(',')),
                                              'location_id'].to_list()
        is_pakistan = data['location_id'].isin(pakistan_location_ids)
        data = data.loc[~is_pakistan].reset_index(drop=True)
        manipulation_metadata['pakistan'] = 'dropped all hospitalizations'
        
        ## ECDC is garbage
        ecdc_location_ids = [77, 82, 83, 59, 60, 88, 91, 52, 55]
        is_ecdc = data['location_id'].isin(ecdc_location_ids)
        data = data.loc[~is_ecdc].reset_index(drop=True)
        manipulation_metadata['ecdc_countries'] = 'dropped all hospitalizations'
        
        ## CLOSE, but seems a little low... check w/ new data
        is_goa = data['location_id'] == 4850
        data = data.loc[~is_goa].reset_index(drop=True)
        manipulation_metadata['goa'] = 'dropped all hospitalizations'

        ## too late, starts March 2021
        is_haiti = data['location_id'] == 114
        data = data.loc[~is_haiti].reset_index(drop=True)
        manipulation_metadata['haiti'] = 'dropped all hospitalizations'

        ## late, starts Jan/Feb 2021 (and is a little low, should check w/ new data)
        is_jordan = data['location_id'] == 144
        data = data.loc[~is_jordan].reset_index(drop=True)
        manipulation_metadata['jordan'] = 'dropped all hospitalizations'
        
        ## too low then too high? odd series
        is_andorra = data['location_id'] == 74
        data = data.loc[~is_andorra].reset_index(drop=True)
        manipulation_metadata['andorra'] = 'dropped all hospitalizations'
    
    elif input_measure == 'deaths':
        pass
    
    else:
        raise ValueError(f'Input measure {input_measure} does not have a protocol for exclusions.')
    
    return data, manipulation_metadata


def seroprevalence(model_inputs_root: Path, verbose: bool = True,) -> pd.DataFrame:
    # load
    data_path = model_inputs_root / 'serology' / 'global_serology_summary.csv'
    data = pd.read_csv(data_path)
    if verbose:
        logger.info(f'Initial observation count: {len(data)}')

    # date formatting
    if 'Date' in data.columns:
        if 'date' in data.columns:
            raise ValueError('Both `Date` and `date` in serology data.')
        else:
            data = data.rename(columns={'Date':'date'})
    for date_var in ['start_date', 'date']:
        # data[date_var] = helpers.str_fmt(data[date_var]).replace('.202$', '.2020')
        # data.loc[(data['location_id'] == 570) & (data[date_var] == '11.08.2021'), date_var] = '11.08.2020'
        # data.loc[(data['location_id'] == 533) & (data[date_var] == '13.11.2.2020'), date_var] = '13.11.2020'
        # data.loc[data[date_var] == '05.21.2020', date_var] = '21.05.2020'
        data[date_var] = pd.to_datetime(data[date_var])  # , format='%d.%m.%Y'
    
    # if no start date provided, assume 2 weeks before end date?
    data['start_date'] = data['start_date'].fillna(data['date'] - pd.Timedelta(days=14))

    # convert to m/l/u to 0-1, sample size to numeric
    if not (helpers.str_fmt(data['units']).unique() == 'percentage').all():
        raise ValueError('Units other than percentage present.')
    data['lower'] = helpers.str_fmt(data['lower']).replace('not specified', np.nan).astype(float)
    data['upper'] = helpers.str_fmt(data['upper']).replace('not specified', np.nan).astype(float)
    data['seroprevalence'] = data['value'] / 100
    data['seroprevalence_lower'] = data['lower'] / 100
    data['seroprevalence_upper'] = data['upper'] / 100
    data['sample_size'] = helpers.str_fmt(data['sample_size']).replace(('unchecked', 'not specified'), np.nan).astype(float)
    
    data['bias'] = helpers.str_fmt(data['bias']).replace(('unchecked', 'not specified'), np.nan).astype(float)
    
    data['test_target'] = helpers.str_fmt(data['test_target']).str.lower()
    
    data['study_start_age'] = helpers.str_fmt(data['study_start_age']).replace(('not specified'), np.nan).astype(float)
    data['study_end_age'] = helpers.str_fmt(data['study_end_age']).replace(('not specified'), np.nan).astype(float)
    
    ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
    ## manually specify certain tests when reporting is mixed (might just be US?)
    # Oxford "mixed" is spike
    is_oxford = data['test_name'] == 'University of Oxford ELISA IgG'
    is_mixed = data['test_target'] == 'mixed'
    data.loc[is_oxford & is_mixed, 'test_target'] = 'spike'
    
    # Peru N-Roche has the wrong isotype
    is_peru = data['location_id'] == 123
    is_roche = data['test_name'] == 'Roche Elecsys N pan-Ig'
    data.loc[is_peru & is_roche, 'isotype'] = 'pan-Ig'
    
    # New York (after Nov 2020 onwards, nucleocapsid test is Abbott, not Roche)
    # ADDENDUM (2021-08-31): mixed portion looks the same as the Abbott; recode that as well
    is_ny = data['location_id'] == 555
    is_cdc = data['survey_series'] == 'cdc_series'
    #is_N = data['test_target'] == 'nucleocapsid'
    is_nov_or_later = data['date'] >= pd.Timestamp('2020-11-01')
    data.loc[is_ny & is_cdc & is_nov_or_later, 'isotype'] = 'pan-Ig'
    data.loc[is_ny & is_cdc & is_nov_or_later, 'test_target'] = 'nucleocapsid'  #  & is_N
    data.loc[is_ny & is_cdc & is_nov_or_later, 'test_name'] = 'Abbott Architect IgG; Roche Elecsys N pan-Ig'  #  & is_N
    
    # Louisiana mixed portion looks the same as the nucleocapsid; recode (will actually use average, see below)
    is_la = data['location_id'] == 541
    is_cdc = data['survey_series'] == 'cdc_series'
    is_nov_or_later = data['date'] >= pd.Timestamp('2020-11-01')
    data.loc[is_la & is_cdc & is_nov_or_later, 'isotype'] = 'pan-Ig'
    data.loc[is_la & is_cdc & is_nov_or_later, 'test_target'] = 'nucleocapsid'
    data.loc[is_la & is_cdc & is_nov_or_later, 'test_name'] = 'Abbott Architect IgG; Roche Elecsys N pan-Ig'
    
    # BIG CDC CHANGE
    # many states are coded as Abbott, seem be Roche after the changes in Nov; recode
    for location_id in [523,  # Alabama
                        526,  # Arkansas
                        527,  # California
                        530,  # Delaware
                        531,  # District of Columbia
                        532,  # Florida
                        536,  # Illinois
                        540,  # Kentucky
                        545,  # Michigan
                        547,  # Mississippi
                        548,  # Missouri
                        551,  # Nevada
                        556,  # North Carolina
                        558,  # Ohio
                        563,  # South Carolina
                        564,  # South Dakota
                        565,  # Tennessee
                        566,  # Texas
                        567,  # Utah
                        571,  # West Virginia
                        572,  # Wisconsin
                        573,  # Wyoming
                       ]:
        is_state = data['location_id'] == location_id
        is_cdc = data['survey_series'] == 'cdc_series'
        is_N = data['test_target'] == 'nucleocapsid'
        is_nov_or_later = data['date'] >= pd.Timestamp('2020-11-01')
        data.loc[is_state & is_cdc & is_nov_or_later & is_N, 'isotype'] = 'pan-Ig'
        data.loc[is_state & is_cdc & is_nov_or_later & is_N, 'test_target'] = 'nucleocapsid'
        data.loc[is_state & is_cdc & is_nov_or_later & is_N, 'test_name'] = 'Roche Elecsys N pan-Ig'  # 'Abbott Architect IgG; Roche Elecsys N pan-Ig'
    ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
    
    outliers = []
    data['manual_outlier'] = data['manual_outlier'].astype(float)
    data['manual_outlier'] = data['manual_outlier'].fillna(0)
    data['manual_outlier'] = data['manual_outlier'].astype(int)
    manual_outlier = data['manual_outlier']
    outliers.append(manual_outlier)
    if verbose:
        logger.info(f'{manual_outlier.sum()} rows from sero data flagged as outliers in ETL.')
    ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
    ## SOME THINGS
    # # 1)
    # #    Question: How to get complete SS?
    # #    Current approach: CI -> SE -> SS where possible; fill with min(SS) where we also don't have CI (very few rows).
    # #    Final solution: ...
    # ss = ss_from_ci(data['seroprevalence'], data['seroprevalence_lower'], data['seroprevalence_upper'])
    # n_missing_ss = (data['sample_size'].isnull() & ss.notnull()).sum()
    # n_missing_ss_ci = (data['sample_size'].isnull() & ss.isnull()).sum()
    # data['sample_size'] = data['sample_size'].fillna(ss)
    # data['sample_size'] = data['sample_size'].fillna(data['sample_size'].min())
    # logger.info(f'Inferring sample size from CI for {n_missing_ss} studies; '
    #             f'filling missing sample size with min observed for {n_missing_ss_ci} that also do not report CI.')
    # del n_missing_ss, n_missing_ss_ci
    
    # # 2)
    # #    Question: What if survey is only in adults? Only kids?
    # #    Current approach: Drop beyond some threshold limits.
    # #    Final solution: ...
    # max_start_age = 30
    # min_end_age = 60
    # data['study_start_age'] = helpers.str_fmt(data['study_start_age']).replace('not specified', np.nan).astype(float)
    # data['study_end_age'] = helpers.str_fmt(data['study_end_age']).replace('not specified', np.nan).astype(float)
    # too_old = data['study_start_age'] > max_start_age
    # too_young = data['study_end_age'] < min_end_age
    # age_outlier = (too_old  | too_young).astype(int)
    # outliers.append(age_outlier)
    # if verbose:
    #     logger.info(f'{age_outlier.sum()} rows from sero data do not have enough '
    #             f'age coverage (at least ages {max_start_age} to {min_end_age}).')
    
    # 3)
    #    Question: Use of geo_accordance?
    #    Current approach: Drop non-represeentative (geo_accordance == 0).
    #    Final solution: ...
    data['geo_accordance'] = helpers.str_fmt(data['geo_accordance']).replace(('unchecked', np.nan), '0').astype(int)
    geo_outlier = data['geo_accordance'] == 0
    outliers.append(geo_outlier)
    if verbose:
        logger.info(f'{geo_outlier.sum()} rows from sero data do not have `geo_accordance`.')
    data['correction_status'] = helpers.str_fmt(data['correction_status']).replace(('unchecked', 'not specified', np.nan), '0').astype(int)
    
    # vaccine debacle, lose all the UK spike data in 2021
    is_uk = data['location_id'].isin([4749, 433, 434, 4636])
    is_spike = data['test_target'] == 'spike'
    is_2021 = data['date'] >= pd.Timestamp('2021-01-01')
    
    uk_vax_outlier = is_uk & is_spike & is_2021
    outliers.append(uk_vax_outlier)
    if verbose:
        logger.info(f'{uk_vax_outlier.sum()} rows from sero data dropped due to UK vax issues.')
        
    # vaccine debacle, lose all the Danish data from Feb 2021 onward
    is_den = data['location_id'].isin([78])
    is_spike = data['test_target'] == 'spike'
    is_2021 = data['date'] >= pd.Timestamp('2021-02-01')
    
    den_vax_outlier = is_den & is_spike & is_2021
    outliers.append(den_vax_outlier)
    if verbose:
        logger.info(f'{den_vax_outlier.sum()} rows from sero data dropped due to Denmark vax issues.')
        
    # vaccine debacle, lose all the Estonia and Netherlands data from Junue 2021 onward
    is_est_ndl = data['location_id'].isin([58, 89])
    is_spike = data['test_target'] == 'spike'
    is_2021 = data['date'] >= pd.Timestamp('2021-06-01')
    
    est_ndl_vax_outlier = is_est_ndl & is_spike & is_2021
    outliers.append(est_ndl_vax_outlier)
    if verbose:
        logger.info(f'{est_ndl_vax_outlier.sum()} rows from sero data dropped due to Netherlands and Estonia vax issues.')
        
    # # drop Rio Grande do Sul
    # rgds_outlier = data['location_id'] == 4772

    # outliers.append(rgds_outlier)
    # if verbose:
    #     logger.info(f'{rgds_outlier.sum()} rows from sero data dropped due to implausible in Rio Grande do Sul.')
    ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##

    keep_columns = ['data_id', 'nid', 'location_id', 'start_date', 'date',
                    'seroprevalence',  # 'sample_size',
                    'study_start_age', 'study_end_age',
                    'test_name', 'test_target', 'isotype',
                    'bias', 'bias_type',
                    'correction_status', 'geo_accordance',
                    'is_outlier', 'manual_outlier']
    data['is_outlier'] = pd.concat(outliers, axis=1).max(axis=1).astype(int)
    data = (data
            .sort_values(['location_id', 'date'])
            .reset_index(drop=True))
    data['data_id'] = data.index
    data = data.loc[:, keep_columns]
    
    if verbose:
        logger.info(f"Final ETL inlier count: {len(data.loc[data['is_outlier'] == 0])}")
        logger.info(f"Final ETL outlier count: {len(data.loc[data['is_outlier'] == 1])}")
    
    return data


def reported_epi(model_inputs_root: Path, input_measure: str,
                 hierarchy: pd.DataFrame, gbd_hierarchy: pd.DataFrame,
                 excess_mortality: bool = None,) -> Tuple[pd.Series, pd.Series]:
    if input_measure == 'deaths':
        if type(excess_mortality) != bool:
            raise TypeError('Must specify `excess_mortality` argument to load deaths.')
        if excess_mortality:
            data_path = model_inputs_root / 'use_at_your_own_risk' / 'full_data_extra_hospital.csv'
        else:
            data_path = model_inputs_root / 'full_data_unscaled.csv'
            logger.info('Using unscaled deaths.')
    else:
        data_path = model_inputs_root / 'use_at_your_own_risk' / 'full_data_extra_hospital.csv'
    data = pd.read_csv(data_path)
    data = data.rename(columns={'Confirmed': 'cumulative_cases',
                                'Hospitalizations': 'cumulative_hospitalizations',
                                'Deaths': 'cumulative_deaths',})
    data['date'] = pd.to_datetime(data['Date'])
    keep_cols = ['location_id', 'date', f'cumulative_{input_measure}']
    data = data.loc[:, keep_cols].dropna()
    data['location_id'] = data['location_id'].astype(int)
    
    data = (data.groupby('location_id', as_index=False)
            .apply(lambda x: helpers.fill_dates(x, [f'cumulative_{input_measure}']))
            .reset_index(drop=True))
    
    data, manipulation_metadata = evil_doings(data, hierarchy, input_measure)
    
    extra_locations = gbd_hierarchy.loc[gbd_hierarchy['most_detailed'] == 1, 'location_id'].to_list()
    extra_locations = [l for l in extra_locations if l not in hierarchy['location_id'].to_list()]
    
    extra_data = data.loc[data['location_id'].isin(extra_locations)].reset_index(drop=True)
    data = helpers.aggregate_data_from_md(data, hierarchy, f'cumulative_{input_measure}')
    data = (data
            .append(extra_data.loc[:, data.columns])
            .sort_values(['location_id', 'date'])
            .reset_index(drop=True))
    
    data[f'daily_{input_measure}'] = (data
                                      .groupby('location_id')[f'cumulative_{input_measure}']
                                      .apply(lambda x: x.diff().fillna(x)))
    data = data.dropna()
    data = (data
            .set_index(['location_id', 'date'])
            .sort_index())
    
    cumulative_data = data[f'cumulative_{input_measure}']
    daily_data = data[f'daily_{input_measure}']

    return cumulative_data, daily_data


def hierarchy(model_inputs_root:Path, hierarchy_type: str = 'covid_modeling') -> pd.DataFrame:
    if hierarchy_type == 'covid_modeling':
        data_path = model_inputs_root / 'locations' / 'modeling_hierarchy.csv'
    elif hierarchy_type == 'covid_covariate':
        data_path = model_inputs_root / 'locations' / 'covariate_with_aggregates_hierarchy.csv'
    elif hierarchy_type == 'covid_gbd':
        data_path = model_inputs_root / 'locations' / 'gbd_analysis_hierarchy.csv'
    
    data = pd.read_csv(data_path)
    data = data.sort_values('sort_order').reset_index(drop=True)
    
    return data


def population(model_inputs_root: Path, by_age: bool = False) -> pd.Series:
    data_path = model_inputs_root / 'output_measures' / 'population' / 'all_populations.csv'
    data = pd.read_csv(data_path)
    is_2019 = data['year_id'] == 2019
    is_bothsex = data['sex_id'] == 3
    if by_age:
        age_metadata = db.age_metadata()    
        data = (data
                .loc[is_2019 & is_bothsex, ['location_id', 'age_group_id', 'population']])
        data = data.merge(age_metadata)
        data = (data
                .set_index(['location_id', 'age_group_years_start', 'age_group_years_end'])
                .sort_index()
                .loc[:, 'population'])
    else:
        is_allage = data['age_group_id'] == 22
        data = (data
                .loc[is_2019 & is_bothsex & is_allage]
                .set_index('location_id')
                .sort_index()
                .loc[:, 'population'])

    return data


def assay_sensitivity(model_inputs_root: Path,) -> pd.DataFrame:
    # TODO: bootstrapping or something to incorporate uncertainty (would need to digitize this portion from Perez-Saez plots)?
    peluso_path = model_inputs_root / 'serology' / 'waning_immunity' / 'peluso_assay_sensitivity.xlsx'
    perez_saez_paths = [
        model_inputs_root / 'serology' / 'waning_immunity' / 'perez-saez_n-roche.xlsx',
        model_inputs_root / 'serology' / 'waning_immunity' / 'perez-saez_rbd-euroimmun.xlsx',
        model_inputs_root / 'serology' / 'waning_immunity' / 'perez-saez_rbd-roche.xlsx',
    ]
    bond_path = model_inputs_root / 'serology' / 'waning_immunity' / 'bond.xlsx'
    muecksch_path = model_inputs_root / 'serology' / 'waning_immunity' / 'muecksch.xlsx'
    lumley_path = model_inputs_root / 'serology' / 'waning_immunity' / 'lumley.xlsx'
    
    ## PELUSO
    peluso = pd.read_excel(peluso_path)
    peluso['t'] = peluso['Time'].apply(lambda x: int(x.split(' ')[0]) * 30)
    peluso = peluso.rename(columns={'mean': 'sensitivity',
                                    'AntigenAndAssay': 'assay',
                                    'Hospitalization_status': 'hospitalization_status',})
    peluso = peluso.loc[:, ['assay', 'hospitalization_status', 't', 'sensitivity']]
    # only need to keep commercial assays
    peluso = peluso.loc[~peluso['assay'].isin(['Neut-Monogram', 'RBD-LIPS', 'RBD-Split Luc',
                                               'RBD-Lum', 'S-Lum', 'N(full)-Lum', 'N-LIPS',
                                               'N(frag)-Lum', 'N-Split Luc'])]
    peluso['source'] = 'Peluso'
    
    ## PEREZ-SAEZ - start at 21 days out
    perez_saez = pd.concat([pd.read_excel(perez_saez_path) for perez_saez_path in perez_saez_paths])
    perez_saez = perez_saez.loc[perez_saez['t'] >= 21]
    perez_saez['t'] -= 21
    perez_saez = pd.concat([
        pd.concat([perez_saez, pd.DataFrame({'hospitalization_status':'Non-hospitalized'}, index=perez_saez.index)], axis=1),
        pd.concat([perez_saez, pd.DataFrame({'hospitalization_status':'Hospitalized'}, index=perez_saez.index)], axis=1)
    ])
    perez_saez['source'] = 'Perez-Saez'
    
    ## BOND - drop 121-150 point, is only 11 people and can't possibly be at 100%; start 21 days out; only keep Abbott
    bond = pd.read_excel(bond_path)
    bond = bond.loc[bond['days since symptom onset'] != '121???150']
    bond['t_start'] = bond['days since symptom onset'].str.split('???').apply(lambda x: int(x[0]))
    bond['t_end'] = bond['days since symptom onset'].str.split('???').apply(lambda x: int(x[1]))
    bond['t'] = bond[['t_start', 't_end']].mean(axis=1)
    for assay in ['N-Abbott', 'S-DiaSorin', 'N-Roche']:
        bond[assay] = bond[assay].str.split('%').apply(lambda x: float(x[0])) / 100
    bond = pd.melt(bond, id_vars='t', value_vars=['N-Abbott', 'S-DiaSorin', 'N-Roche'],
                   var_name='assay', value_name='sensitivity')
    bond = bond.loc[bond['t'] >= 21]
    bond['t'] -= 21
    bond = pd.concat([
        pd.concat([bond, pd.DataFrame({'hospitalization_status':'Non-hospitalized'}, index=bond.index)], axis=1),
        pd.concat([bond, pd.DataFrame({'hospitalization_status':'Hospitalized'}, index=bond.index)], axis=1)
    ])
    bond = bond.loc[bond['assay'] == 'N-Abbott']
    bond['source'] = 'Bond'
    
    ## MUECKSCH - top end of terminal group is 110 days; only keep Abbott
    muecksch = pd.read_excel(muecksch_path)
    muecksch.loc[muecksch['Time, d'] == '>81', 'Time, d'] = '81-110'
    muecksch['t_start'] = muecksch['Time, d'].str.split('-').apply(lambda x: int(x[0]))
    muecksch['t_end'] = muecksch['Time, d'].str.split('-').apply(lambda x: int(x[1]))
    muecksch['t'] = muecksch[['t_start', 't_end']].mean(axis=1)
    for assay in ['N-Abbott', 'S-DiaSorin', 'RBD-Siemens']:
        muecksch[assay] = muecksch[assay].str.split(' ').apply(lambda x: float(x[0])) / 100
    muecksch = pd.melt(muecksch, id_vars='t', value_vars=['N-Abbott', 'S-DiaSorin', 'RBD-Siemens'],
                       var_name='assay', value_name='sensitivity')
    muecksch['t'] -= 24
    muecksch = pd.concat([
        pd.concat([muecksch, pd.DataFrame({'hospitalization_status':'Non-hospitalized'}, index=muecksch.index)], axis=1),
        pd.concat([muecksch, pd.DataFrame({'hospitalization_status':'Hospitalized'}, index=muecksch.index)], axis=1)
    ])
    muecksch = muecksch.loc[muecksch['assay'] == 'N-Abbott']
    muecksch['source'] = 'Muecksch'
    
    ## LUMLEY
    lumley = pd.read_excel(lumley_path)
    lumley = lumley.loc[lumley['keep'] == 1]
    lumley['sensitivity'] *= (lumley['num_60'] / lumley['denom_60']) / lumley['avg_60']
    lumley = pd.concat([
        pd.concat([lumley, pd.DataFrame({'hospitalization_status':'Non-hospitalized'}, index=lumley.index)], axis=1),
        pd.concat([lumley, pd.DataFrame({'hospitalization_status':'Hospitalized'}, index=lumley.index)], axis=1)
    ])
    lumley['source'] = 'Lumley'
    
    # combine them all
    keep_cols = ['source', 'assay', 'hospitalization_status', 't', 'sensitivity',]
    sensitivity = pd.concat([peluso.loc[:, keep_cols],
                             perez_saez.loc[:, keep_cols],
                             bond.loc[:, keep_cols],
                             muecksch.loc[:, keep_cols],
                             lumley.loc[:, keep_cols],]).reset_index(drop=True)
    
    return sensitivity


def assay_map(model_inputs_root: Path,):
    data_path = model_inputs_root / 'serology' / 'waning_immunity' / 'assay_map.xlsx'
    data = pd.read_excel(data_path)
    
    return data


def validate_hierarchies(hierarchy: pd.DataFrame, gbd_hierarchy: pd.DataFrame):
    covid = hierarchy.loc[:, ['location_id', 'path_to_top_parent']]
    covid = covid.rename(columns={'path_to_top_parent': 'covid_path'})
    gbd = gbd_hierarchy.loc[:, ['location_id', 'path_to_top_parent']]
    gbd = gbd.rename(columns={'path_to_top_parent': 'gbd_path'})
    
    data = covid.merge(gbd, how='left')
    is_missing = data['gbd_path'].isnull()
    is_different = (data['covid_path'] != data['gbd_path']) & (~is_missing)
    
    if is_different.sum() > 0:
        raise ValueError(f'Some covid locations are missing a GBD path:\n{data.loc[is_different]}.')
    
    if is_missing.sum() > 0:
        logger.warning(f'Some covid locations are missing in GBD hierarchy and will be added:\n{data.loc[is_missing]}.')
        missing_locations = data.loc[is_missing, 'location_id'].to_list()
        missing_locations = hierarchy.loc[hierarchy['location_id'].isin(missing_locations)]
        gbd_hierarchy = gbd_hierarchy.append(missing_locations)
        
    return gbd_hierarchy
