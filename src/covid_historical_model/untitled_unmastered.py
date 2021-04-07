from pathlib import Path
import yaml
import dill as pickle

import pandas as pd

from covid_shared import cli_tools, shell_tools

from covid_historical_model.rates.pipeline import pipeline
from covid_historical_model.durations.durations import EXPOSURE_TO_SEROPOSITIVE

import warnings
warnings.simplefilter('ignore')

#######################################
#### TRY TRIMMING IN GLOBAL MODEL? ####
#######################################

## IMPORTANT TODO:
##     - reinfection NAs (probably 0 deaths)
##     - best way to fill where we have no assay information
##     - checks on EM file
##     - bias covariate?
##     - for waning, do something to Perez-Saez to crosswalk for baseline sensitivity?
##     - NAs in IES inputs?

## RATIO FUTURE TODO:
##     - try trimming in certain levels (probably just global)? might screw up spline, might not
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
##     - splines
##          (a) try higher degree, fewer knot-days?
##          (b) would it work to fix them e.g. every month?
##          (c) "fix" uncertainty

def main(app_metadata: cli_tools.Metadata, out_dir: Path,
         model_inputs_root: Path,
         vaccine_coverage_root: Path, variant_scaleup_root: Path,
         age_pattern_root: Path, testing_root: Path,
         excess_mortality: bool,):
    ## working dir
    storage_dir = out_dir / 'intermediate'
    results_dir = out_dir / 'results'
    plots_dir = out_dir / 'plots'
    shell_tools.mkdir(storage_dir)
    shell_tools.mkdir(results_dir)
    shell_tools.mkdir(plots_dir)

    ## run models
    seroprevalence, reinfection_inflation_factor, ifr_results, idr_results, ihr_results, em_data = pipeline(
        out_dir, storage_dir, plots_dir,
        model_inputs_root, excess_mortality,
        vaccine_coverage_root, variant_scaleup_root,
        age_pattern_root,
        testing_root,
    )
    with (results_dir / 'ifr_results.pkl').open('wb') as file:
        pickle.dump(ifr_results, file, -1)
    with (results_dir / 'idr_results.pkl').open('wb') as file:
        pickle.dump(idr_results, file, -1)
    with (results_dir / 'ihr_results.pkl').open('wb') as file:
        pickle.dump(ihr_results, file, -1)

    ## save IFR
    ifr = pd.concat([ifr_results.pred.rename('ifr'),
                     ifr_results.pred_fe.rename('ifr_no_random_effect'),],
                    axis=1)
    ifr = ifr.reset_index()

    ifr_risk_adjustment = pd.concat([ifr_results.pred.rename('ifr'),
                                     ifr_results.pred_lr.rename('ifr_lr'),
                                     ifr_results.pred_hr.rename('ifr_hr'),],
                                    axis=1)
    ifr_risk_adjustment = ifr_risk_adjustment.reset_index()

    ifr_data = ifr_results.model_data.copy()
    del ifr_data['date']
    ifr_data = ifr_data.rename(columns={'mean_death_date':'date'})
    ifr_data['is_outlier'] = 0
    ifr_data = ifr_data.loc[:, ['location_id', 'date', 'ifr', 'is_outlier']]

    ## save IHR
    ihr = pd.concat([ihr_results.pred.rename('ihr'),
                     ihr_results.pred_fe.rename('ihr_no_random_effect'),],
                    axis=1)
    ihr = ihr.reset_index()

    ihr_data = ihr_results.model_data.copy()
    del ihr_data['date']
    ihr_data = ihr_data.rename(columns={'mean_hospitalization_date':'date'})
    ihr_data['is_outlier'] = 0
    ihr_data = ihr_data.loc[:, ['location_id', 'date', 'ihr', 'is_outlier']]

    ## save IDR
    idr = pd.concat([idr_results.pred.rename('idr'),
                     idr_results.pred_fe.rename('idr_fe'),],
                    axis=1)
    idr = idr.reset_index()

    idr_data = idr_results.model_data.copy()
    idr_data = idr_data.rename(columns={'mean_death_date':'date'})
    idr_data['is_outlier'] = 0
    idr_data = idr_data.loc[:, ['location_id', 'date', 'idr', 'is_outlier']]

    ## save serology
    seroprevalence = (seroprevalence
                      .loc[:, ['data_id', 'location_id', 'date', 'geo_accordance', 'manual_outlier',
                               'reported_seroprevalence','seroprevalence']])
    seroprevalence = seroprevalence.rename(columns={'seroprevalence':'seroprev_mean_no_vacc',
                                                    'reported_seroprevalence':'seroprev_mean'})
    seroprevalence = (seroprevalence
                      .merge(ifr_results.seroprevalence.loc[:, ['data_id', 'location_id', 'date', 'seroprevalence']]
                             .rename(columns={'seroprevalence':'seroprev_mean_no_vacc_waning'}),
                    how='left'))
    seroprevalence['infection_date'] = seroprevalence['date'] - pd.Timedelta(days=EXPOSURE_TO_SEROPOSITIVE)
    del seroprevalence['date']

    ## save testing placeholder
    testing = pd.DataFrame({'location_id': [],
                            'date': [],
                            'testing': []})

    ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
    ## write outputs
    em_data.to_csv(out_dir / 'excess_mortalilty.csv', index=False)

    ifr.to_csv(out_dir / 'allage_ifr_by_loctime.csv', index=False)
    ifr_risk_adjustment.to_csv(out_dir / 'terminal_ifr.csv', index=False)
    ifr_data.to_csv(out_dir / 'ifr_model_data.csv', index=False)

    ihr.to_csv(out_dir / 'allage_ihr_by_loctime.csv', index=False)
    ihr_data.to_csv(out_dir / 'ihr_model_data.csv', index=False)

    idr.to_csv(out_dir / 'pred_idr.csv', index=False)
    idr_data.to_csv(out_dir / 'idr_plot_data.csv', index=False)

    seroprevalence.to_csv(out_dir / 'sero_data.csv', index=False)

    reinfection_inflation_factor.reset_index().to_csv(out_dir / 'reinfection_data.csv', index=False)

    testing.to_csv(out_dir / 'test_data.csv', index=False)

    with open(out_dir / 'metadata.yaml', 'w') as file:
        yaml.dump({}, file)
