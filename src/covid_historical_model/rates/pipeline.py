import sys
from typing import List, Tuple, Dict
from pathlib import Path
import itertools
from loguru import logger
import dill as pickle
import shutil

import pandas as pd
import numpy as np

from covid_shared import shell_tools
from covid_shared.cli_tools.logging import configure_logging_to_terminal

from covid_historical_model.etl import model_inputs, estimates, db
from covid_historical_model.rates import serology
from covid_historical_model.rates import covariate_selection
from covid_historical_model.durations import durations
from covid_historical_model.rates import age_standardization
from covid_historical_model.rates import ifr
from covid_historical_model.rates import idr
from covid_historical_model.rates import ihr
from covid_historical_model.rates import cvi
from covid_historical_model import cluster
from covid_historical_model.rates import location_plots
from covid_historical_model.utils import pdf_merger


def pipeline_wrapper(out_dir: Path,
                     model_inputs_root: Path, excess_mortality: bool,
                     vaccine_coverage_root: Path, variant_scaleup_root: Path,
                     age_rates_root: Path,
                     testing_root: Path,
                     n_samples: int,
                     day_inflection_options: List[str] = ['2020-05-01', '2020-06-01', '2020-07-01',
                                                          '2020-08-01', '2020-09-01', '2020-10-01',
                                                          '2020-11-01', '2020-12-01', '2021-01-01',
                                                          '2021-02-01', '2021-03-01',],
                     correlate_samples: bool = True,
                     bootstrap: bool = True,
                     verbose: bool = True,) -> Tuple:
    np.random.seed(15243)
    if verbose:
        logger.info('Loading variant, vaccine, and sero data.')
    hierarchy = model_inputs.hierarchy(model_inputs_root)
    gbd_hierarchy = model_inputs.hierarchy(model_inputs_root, 'covid_gbd')
    adj_gbd_hierarchy = model_inputs.validate_hierarchies(hierarchy.copy(), gbd_hierarchy.copy())
    population = model_inputs.population(model_inputs_root)
    age_spec_population = model_inputs.population(model_inputs_root, by_age=True)
    population_lr, population_hr = age_standardization.get_risk_group_populations(age_spec_population)
    shared = {
        'hierarchy': hierarchy,
        'gbd_hierarchy': gbd_hierarchy,
        'adj_gbd_hierarchy': adj_gbd_hierarchy,
        'population': population,
        'age_spec_population': age_spec_population,
        'population_lr': population_lr,
        'population_hr': population_hr,
    }
    
    escape_variant_prevalence = estimates.variant_scaleup(variant_scaleup_root, 'escape', verbose=verbose)
    severity_variant_prevalence = estimates.variant_scaleup(variant_scaleup_root, 'severity', verbose=verbose)
    vaccine_coverage = estimates.vaccine_coverage(vaccine_coverage_root)
    reported_seroprevalence, seroprevalence_samples = serology.load_seroprevalence_sub_vacccinated(
        model_inputs_root, vaccine_coverage.copy(), n_samples=n_samples,
        correlate_samples=correlate_samples, bootstrap=bootstrap,
        verbose=verbose,
    )
    reported_sensitivity_data, sensitivity_data_samples = serology.load_sensitivity(model_inputs_root, n_samples,)
    cross_variant_immunity_samples = cvi.get_cvi_dist(n_samples)
    durations_samples = durations.get_duration_dist(n_samples)
    
    covariate_options = ['obesity', 'smoking', 'diabetes', 'ckd',
                         'cancer', 'copd', 'cvd', 'uhc', 'haq',]
    covariates = [db.obesity(adj_gbd_hierarchy),
                  db.smoking(adj_gbd_hierarchy),
                  db.diabetes(adj_gbd_hierarchy),
                  db.ckd(adj_gbd_hierarchy),
                  db.cancer(adj_gbd_hierarchy),
                  db.copd(adj_gbd_hierarchy),
                  db.cvd(adj_gbd_hierarchy),
                  db.uhc(adj_gbd_hierarchy) / 100,
                  db.haq(adj_gbd_hierarchy) / 100,]
    
    if verbose:
        logger.info('Identifying best covariate combinations and creating input data object.')
    # for now, just make up covariates
    test_combinations = []
    for i in range(len(covariate_options)):
        test_combinations += [list(set(cc)) for cc in itertools.combinations(covariate_options, i + 1)]
    test_combinations = [cc for cc in test_combinations if 
                        len([c for c in cc if c in ['uhc', 'haq']]) <= 1]
    selected_combinations = covariate_selection.covariate_selection(
        n_samples=n_samples, test_combinations=test_combinations,
        model_inputs_root=model_inputs_root, excess_mortality=excess_mortality,
        age_rates_root=age_rates_root, shared=shared,
        reported_seroprevalence=reported_seroprevalence,
        covariate_options=covariate_options,
        covariates=covariates,
        cutoff_pct=1.,
        durations={'sero_to_death': int(round(np.mean(durations.EXPOSURE_TO_ADMISSION) + \
                                              np.mean(durations.ADMISSION_TO_DEATH) - \
                                              np.mean(durations.EXPOSURE_TO_SEROCONVERSION)))
                  },
    )
    idr_covariate_pool = np.random.choice([['haq'], ['uhc'], ['prop_65plus'], []], n_samples)
    day_inflection_pool = np.random.choice(day_inflection_options, n_samples)
    day_inflection_pool = [str(d) for d in day_inflection_pool]  # can't be np.str_
    
    inputs = {
        n: {
            'orig_seroprevalence': seroprevalence,
            'shared': shared,
            'model_inputs_root': model_inputs_root,
            'excess_mortality': excess_mortality,
            'sensitivity_data': sensitivity_data,
            'vaccine_coverage': vaccine_coverage,
            'escape_variant_prevalence': escape_variant_prevalence,
            'severity_variant_prevalence': severity_variant_prevalence,
            'age_rates_root': age_rates_root,
            'testing_root': testing_root,
            'day_inflection': day_inflection,
            'covariates': covariates,
            'covariate_list': covariate_list,
            'idr_covariate_list': idr_covariate_list,
            'cross_variant_immunity': cross_variant_immunity,
            'durations': durations,
            'verbose': verbose,
        }
        for n, (covariate_list, idr_covariate_list,
                seroprevalence, sensitivity_data,
                cross_variant_immunity, day_inflection, durations,)
        in enumerate(zip(selected_combinations, idr_covariate_pool,
                         seroprevalence_samples, sensitivity_data_samples,
                         cross_variant_immunity_samples, day_inflection_pool, durations_samples,))
    }
    
    if verbose:
        logger.info('Storing inputs and submitting sero-sample jobs.')
    inputs_path = out_dir / 'pipeline_inputs.pkl'
    with inputs_path.open('wb') as file:
        pickle.dump(inputs, file, -1)
    pipeline_dir = out_dir / 'pipeline_outputs'
    shell_tools.mkdir(pipeline_dir)
    job_args_map = {n: [__file__, n, inputs_path, pipeline_dir] for n in range(n_samples)}
    cluster.run_cluster_jobs('covid_rates_pipeline', out_dir, job_args_map)
    
    pipeline_results = {}
    for n in range(n_samples):
        with (pipeline_dir / f'{n}_outputs.pkl').open('rb') as file:
            outputs = pickle.load(file)
        pipeline_results.update(outputs)
    
    em_data = estimates.excess_mortailty_scalars(model_inputs_root, excess_mortality)
    
    return pipeline_results, selected_combinations, cross_variant_immunity_samples, \
           reported_seroprevalence, reported_sensitivity_data, \
           escape_variant_prevalence, severity_variant_prevalence, \
           vaccine_coverage, em_data


def pipeline(orig_seroprevalence: pd.DataFrame,
             shared: Dict,
             model_inputs_root: Path, excess_mortality: bool,
             sensitivity_data: pd.DataFrame,
             vaccine_coverage: pd.DataFrame,
             escape_variant_prevalence: pd.Series,
             severity_variant_prevalence: pd.Series,
             age_rates_root: Path,
             testing_root: Path,
             day_inflection: str,
             covariates: List[pd.Series],
             covariate_list: List[str],
             idr_covariate_list: List[str],
             cross_variant_immunity: float,
             durations: Dict,
             verbose: bool,) -> Tuple:
    if verbose:
        logger.info('\n*************************************\n'
                    f"IFR ESTIMATION -- inflection point at {day_inflection}\n"
                    '*************************************')
    ifr_input_data = ifr.data.load_input_data(model_inputs_root, excess_mortality,
                                              age_rates_root,
                                              shared.copy(),
                                              orig_seroprevalence.copy(), sensitivity_data.copy(), vaccine_coverage.copy(),
                                              escape_variant_prevalence.copy(), severity_variant_prevalence.copy(),
                                              covariates.copy(),
                                              cross_variant_immunity,
                                              verbose=verbose)
    # , daily_reinfection_inflation_factor
    ifr_results, adj_seroprevalence, sensitivity, \
    cumul_reinfection_inflation_factor, _ = ifr.runner.runner(
        input_data=ifr_input_data,
        day_inflection=day_inflection,
        covariate_list=covariate_list,
        durations=durations,
        verbose=verbose,
    )

    if verbose:
        logger.info('\n*************************************\n'
                    'IDR ESTIMATION\n'
                    '*************************************')
    
    idr_input_data = idr.data.load_input_data(model_inputs_root, excess_mortality, testing_root,
                                              shared.copy(),
                                              adj_seroprevalence.copy(), vaccine_coverage.copy(),
                                              escape_variant_prevalence.copy(),
                                              covariates.copy(),
                                              cross_variant_immunity,
                                              verbose=verbose)
    idr_results = idr.runner.runner(idr_input_data,
                                    ifr_results.pred.copy(),
                                    idr_covariate_list,
                                    durations,
                                    verbose=verbose)
    
    if verbose:
        logger.info('\n*************************************\n'
                    'IHR ESTIMATION\n'
                    '*************************************')
    ihr_input_data = ihr.data.load_input_data(model_inputs_root,
                                              age_rates_root,
                                              shared.copy(),
                                              adj_seroprevalence.copy(), vaccine_coverage.copy(),
                                              escape_variant_prevalence.copy(), severity_variant_prevalence.copy(),
                                              covariates.copy(),
                                              cross_variant_immunity,
                                              verbose=verbose)
    ihr_results = ihr.runner.runner(ihr_input_data, covariate_list, durations,
                                    verbose=verbose)
    
    if verbose:
        logger.info('\n*************************************\n'
                    'PIPELINE COMPLETE -- preparing storage and saving results \n'
                    '*************************************')
    pipeline_results = {
        'covariate_list': covariate_list,
        'idr_covariate_list': idr_covariate_list,
        'durations': durations,
        'seroprevalence': adj_seroprevalence,
        'sensitivity': sensitivity,
        'cumul_reinfection_inflation_factor': cumul_reinfection_inflation_factor,
        # 'daily_reinfection_inflation_factor': daily_reinfection_inflation_factor,
        'day_inflection': day_inflection,
        'ifr_results': ifr_results,
        'idr_results': idr_results,
        'ihr_results': ihr_results,
    }
    
    return pipeline_results


def compile_pdfs(plots_dir: Path, out_dir: Path, hierarchy: pd.DataFrame,
                 outfile_prefix: str, suffixes: List[str],):
    possible_pdfs = [[f'{l}_{s}.pdf' for s in suffixes] for l in hierarchy['location_id']]
    possible_pdfs = [ll for l in possible_pdfs for ll in l]
    existing_pdfs = [str(x).split('/')[-1] for x in plots_dir.iterdir() if x.is_file()]
    pdf_paths = [pdf for pdf in possible_pdfs if pdf in existing_pdfs]
    pdf_location_ids = [int(pdf_path.split('_')[0]) for pdf_path in pdf_paths]
    pdf_location_names = [hierarchy.loc[hierarchy['location_id'] == location_id, 'location_name'].item()
                          for location_id in pdf_location_ids]
    pdf_parent_ids = [hierarchy.loc[hierarchy['location_id'] == location_id, 'parent_id'].item()
                      for location_id in pdf_location_ids]
    pdf_parent_names = [hierarchy.loc[hierarchy['location_id'] == parent_id, 'location_name'].item()
                        for parent_id in pdf_parent_ids]
    pdf_levels = [hierarchy.loc[hierarchy['location_id'] == location_id, 'level'].item()
                  for location_id in pdf_location_ids]
    pdf_paths = [str(plots_dir / pdf_path) for pdf_path in pdf_paths]
    pdf_out_path = out_dir / f'{outfile_prefix}_{str(out_dir).split("/")[-1]}.pdf'
    pdf_merger.pdf_merger(pdf_paths, pdf_location_names, pdf_parent_names, pdf_levels, str(pdf_out_path))

    
def submit_plots():
    raise ValueError('PLOTTING NOT SET UP')
    plot_inputs = {
        'best_ifr_models': best_ifr_models.set_index('location_id'),
        'seroprevalence': seroprevalence,
        'sensitivity': sensitivity,
        'sensitivity_data': sensitivity_data,
        'cumul_reinfection_inflation_factor': cumul_reinfection_inflation_factor,
        'daily_reinfection_inflation_factor': daily_reinfection_inflation_factor,
        'full_ifr_results': full_ifr_results,
        'ifr_results': ifr_results,
        'vaccine_coverage': vaccine_coverage,
        'escape_variant_prevalence': escape_variant_prevalence,
        'severity_variant_prevalence': severity_variant_prevalence,
        'population': population,
        'population_lr': population_lr,
        'population_hr': population_hr,
        'hierarchy': hierarchy,
    }
    inputs_path = storage_dir / f'plot_inputs.pkl'
    with inputs_path.open('wb') as file:
        pickle.dump(plot_inputs, file, -1)
    
    job_args_map = {
        location_id: [location_plots.__file__, location_id, inputs_path, plots_dir] \
        for location_id in hierarchy['location_id'].to_list()
    }
    cluster.run_cluster_jobs('covid_rates_plot', storage_dir, job_args_map)
    compile_pdfs(plots_dir, out_dir, hierarchy, 'ifr', suffixes=['ifr'])
    compile_pdfs(plots_dir, out_dir, hierarchy, 'serology', suffixes=['sero'])


def main(n: int, inputs_path: str, pipeline_dir: str):
    with Path(inputs_path).open('rb') as file:
        inputs = pickle.load(file)[n]
    
    np.random.seed(123 * (n + 1))
    pipeline_outputs = pipeline(**inputs)
    
    with (Path(pipeline_dir) / f'{n}_outputs.pkl').open('wb') as file:
        pickle.dump({n: pipeline_outputs}, file)
    

if __name__ == '__main__':
    configure_logging_to_terminal(verbose=2)
    
    main(int(sys.argv[1]), sys.argv[2], sys.argv[3])
