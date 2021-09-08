from pathlib import Path
import dill as pickle
from loguru import logger

import pandas as pd

from covid_shared import cli_tools

from covid_historical_model.rates.pipeline import pipeline_wrapper
from covid_historical_model.durations.durations import EXPOSURE_TO_SEROPOSITIVE

## IMPORTANT TODO:
##     - use date midpoint
##     - make comparison routine; plot all fits in cascade
##     - multiple locations after July 1 for date selection (currently just 1)? unless only one child?
##     - reinfection NAs (probably 0 deaths) -> add checks for location/date matching
##     - other NAs in IES inputs?
##     - best way to fill where we have no assay information
##     - bias covariates?
##     - for waning, do something to Perez-Saez to crosswalk for baseline sensitivity?
##     - smarter posterior IFR forecast
##     - 

## RATIO FUTURE TODO:
##     - try trimming in certain levels (probably just global)?
##     - slope in IHR?
##     - log cluster jobs
##     - make sure we don't have NAs on dates that matter for ratios
##     - make text matching in serology data more robust (e.g. to spelling mistakes)
##     - formalize test matching in `serology.apply_waning_adjustment`
##     - stuff written down in IHME notebook
##     - think through ...
##          (a) how final models are selected for IFR (namely, anything undesirable wrt parent models)
##          (b) is sero data inconsistent between IFR and IHR/IDR?
##     - existing to-do's in IDR model
##     - use fit to find tests where we have multiple? would be a little harder...
##     - mark model data NAs as outliers, drop that way (in general, make it clear what data is and is not included)
##     - remove unused model data in runner after modeling

## JEFFREY FUTURE TODO:
##     - add smarter logic around dropping leading 0s
##     - plot dropped data
##     - remove offset at the end
##     - splines
##          (a) try higher degree, fewer knot-days?
##          (b) would it work to fix them e.g. every month?
##          (c) "fix" uncertainty (maybe the measure cascade)
##     - issue w/ low days pulling down composite (e.g. Connecticut)


def main(app_metadata: cli_tools.Metadata, out_dir: Path,
         model_inputs_root: Path,
         vaccine_coverage_root: Path, variant_scaleup_root: Path,
         age_pattern_root: Path, testing_root: Path,
         excess_mortality: bool,
         n_samples: int,):
    ## run models
    pipeline_results, shared, reported_seroprevalence, sensitivity_data, \
    escape_variant_prevalence, severity_variant_prevalence, vaccine_coverage, em_data = pipeline_wrapper(
        out_dir,
        model_inputs_root, excess_mortality,
        vaccine_coverage_root, variant_scaleup_root,
        age_pattern_root,
        testing_root,
        n_samples,
    )
    
    ## save models
    with (out_dir / 'pipeline_results.pkl').open('wb') as file:
        pickle.dump(pipeline_results, file, -1)
    
    ## save IFR
    ifr_draws = pd.concat([pipeline_results[n]['ifr_results'].pred.rename(f'draw_{n}') for n in range(n_samples)], axis=1)
    ifr_draws = ifr_draws.reset_index()
    ifr_lr_draws = pd.concat([pipeline_results[n]['ifr_results'].pred_lr.rename(f'draw_{n}') for n in range(n_samples)], axis=1)
    ifr_lr_draws = ifr_lr_draws.reset_index()
    ifr_hr_draws = pd.concat([pipeline_results[n]['ifr_results'].pred_hr.rename(f'draw_{n}') for n in range(n_samples)], axis=1)
    ifr_hr_draws = ifr_hr_draws.reset_index()
    ifr_global_draws = pd.concat([pipeline_results[n]['ifr_results'].pred_fe.rename(f'draw_{n}') for n in range(n_samples)], axis=1)
    ifr_global_draws = ifr_global_draws.reset_index()
    
    ifr_model_data_draws = []
    for ifr_model_data in [pipeline_results[n]['ifr_results'].model_data.copy() for n in range(n_samples)]:
        del ifr_model_data['date']
        ifr_model_data = ifr_model_data.rename(columns={'mean_death_date':'date'})
        ifr_model_data['draw'] = n
        ifr_model_data['is_outlier'] = 0
        ifr_model_data = ifr_model_data.loc[:, ['location_id', 'date', 'draw', 'ifr', 'is_outlier']]
        ifr_model_data_draws.append(ifr_model_data)
    ifr_model_data_draws = pd.concat(ifr_model_data_draws).reset_index(drop=True)
    
    ifr_age_stand = pipeline_results[0]['ifr_results'].age_stand_scaling_factor.reset_index()
    
    ifr_level_lambdas = pd.DataFrame(pipeline_results[0]['ifr_results'].level_lambdas).T
    ifr_level_lambdas.index.name = 'hierarchy_level'
    ifr_level_lambdas = ifr_level_lambdas.reset_index()

    ## save IHR
    ihr_draws = pd.concat([pipeline_results[n]['ihr_results'].pred.rename(f'draw_{n}') for n in range(n_samples)], axis=1)
    ihr_draws = ihr_draws.reset_index()
    ihr_lr_draws = pd.concat([pipeline_results[n]['ihr_results'].pred_lr.rename(f'draw_{n}') for n in range(n_samples)], axis=1)
    ihr_lr_draws = ihr_lr_draws.reset_index()
    ihr_hr_draws = pd.concat([pipeline_results[n]['ihr_results'].pred_hr.rename(f'draw_{n}') for n in range(n_samples)], axis=1)
    ihr_hr_draws = ihr_hr_draws.reset_index()
    ihr_global_draws = pd.concat([pipeline_results[n]['ihr_results'].pred_fe.rename(f'draw_{n}') for n in range(n_samples)], axis=1)
    ihr_global_draws = ihr_global_draws.reset_index()

    ihr_model_data_draws = []
    for ihr_model_data in [pipeline_results[n]['ihr_results'].model_data.copy() for n in range(n_samples)]:
        del ihr_model_data['date']
        ihr_model_data = ihr_model_data.rename(columns={'mean_hospitalization_date':'date'})
        ihr_model_data['draw'] = n
        ihr_model_data['is_outlier'] = 0
        ihr_model_data = ihr_model_data.loc[:, ['location_id', 'date', 'draw', 'ihr', 'is_outlier']]
    ihr_model_data_draws = pd.concat(ihr_model_data_draws).reset_index(drop=True)
    
    ihr_age_stand = pipeline_results[0]['ihr_results'].age_stand_scaling_factor.reset_index()
    
    ihr_level_lambdas = pd.DataFrame(pipeline_results[0]['ihr_results'].level_lambdas).T
    ihr_level_lambdas.index.name = 'hierarchy_level'
    ihr_level_lambdas = ihr_level_lambdas.reset_index()

    ## save IDR
    idr_draws = pd.concat([pipeline_results[n]['idr_results'].pred.rename(f'draw_{n}') for n in range(n_samples)], axis=1)
    idr_draws = idr_draws.reset_index()
    idr_global_draws = pd.concat([pipeline_results[n]['idr_results'].pred_fe.rename(f'draw_{n}') for n in range(n_samples)], axis=1)
    idr_global_draws = idr_global_draws.reset_index()

    idr_model_data_draws = []
    for idr_model_data in [pipeline_results[n]['idr_results'].model_data.copy() for n in range(n_samples)]:
        del idr_model_data['date']
        idr_model_data = idr_model_data.rename(columns={'avg_date_of_infection':'date'})
        idr_model_data['draw'] = n
        idr_model_data['is_outlier'] = 0
        idr_model_data = idr_model_data.loc[:, ['location_id', 'date', 'draw', 'idr', 'is_outlier']]
    idr_model_data_draws = pd.concat(idr_model_data_draws).reset_ndex(drop=True)
    
    idr_level_lambdas = pd.DataFrame(pipeline_results[0]['idr_results'].level_lambdas).T
    idr_level_lambdas.index.name = 'hierarchy_level'
    idr_level_lambdas = idr_level_lambdas.reset_index()

    raise ValueError('Remaining: \n    -format sero\n    -save draw files\n    -take this time to improve storage structure')
    
    ## save serology
    seroprevalence = seroprevalence.rename(columns={'seroprevalence':'seroprev_mean_no_vacc',
                                                    'reported_seroprevalence':'seroprev_mean'})
    seroprevalence = (seroprevalence
                      .merge(ifr_results.seroprevalence.loc[:, ['data_id', 'location_id', 'date', 'seroprevalence']]
                             .rename(columns={'seroprevalence':'seroprev_mean_no_vacc_waning'}),
                    how='left'))
    seroprevalence['infection_date'] = seroprevalence['date'] - pd.Timedelta(days=EXPOSURE_TO_SEROPOSITIVE)
    seroprevalence = seroprevalence.rename(columns={'start_date': 'sero_start_date',
                                                    'date': 'sero_end_date'})

    testing = idr_results.testing_capacity.reset_index()

    ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
    ## write outputs
    hierarchy.to_csv(out_dir / 'hierarchy.csv', index=False)
    population.reset_index().to_csv(out_dir / 'population.csv', index=False)
    
    em_data.to_csv(out_dir / 'excess_mortality.csv', index=False)

    ifr.to_csv(out_dir / 'allage_ifr_by_loctime.csv', index=False)
    ifr_risk_adjustment.to_csv(out_dir / 'terminal_ifr.csv', index=False)
    ifr_data.to_csv(out_dir / 'ifr_model_data.csv', index=False)
    ifr_age_stand.to_csv(out_dir / 'ifr_age_stand_data.csv', index=False)
    ifr_level_lambdas.to_csv(out_dir / 'ifr_level_lambdas.csv', index=False)
    ifr_nrmse.to_csv(out_dir / 'ifr_nrmse.csv', index=False)
    best_ifr_models.to_csv(out_dir / 'best_ifr_models.csv', index=False)

    ihr.to_csv(out_dir / 'allage_ihr_by_loctime.csv', index=False)
    ihr_data.to_csv(out_dir / 'ihr_model_data.csv', index=False)
    ihr_age_stand.to_csv(out_dir / 'ihr_age_stand_data.csv', index=False)
    ihr_level_lambdas.to_csv(out_dir / 'ihr_level_lambdas.csv', index=False)

    idr.to_csv(out_dir / 'pred_idr.csv', index=False)
    idr_data.to_csv(out_dir / 'idr_plot_data.csv', index=False)
    idr_level_lambdas.to_csv(out_dir / 'idr_level_lambdas.csv', index=False)

    seroprevalence.to_csv(out_dir / 'sero_data.csv', index=False)

    reinfection_inflation_factor.to_csv(out_dir / 'reinfection_data.csv', index=False)

    testing.to_csv(out_dir / 'test_data.csv', index=False)
    
    vaccine_coverage.reset_index().to_csv(out_dir / 'vaccine_coverage.csv', index=False)
    
    (pd.concat([escape_variant_prevalence, severity_variant_prevalence], axis=1)
     .reset_index()).to_csv(out_dir / 'variants.csv', index=False)

    logger.info(f'Model output directory: {str(out_dir)}')
