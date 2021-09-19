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
                     age_pattern_root: Path, testing_root: Path,
                     n_samples: int,
                     day_inflection_list: List[str] = ['2020-05-01', '2020-06-01', '2020-07-01', '2020-08-01',
                                                       '2020-09-01', '2020-10-01', '2020-11-01', '2020-12-01',],
                     correlate_samples: bool = False,
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
        age_pattern_root=age_pattern_root, shared=shared,
        reported_seroprevalence=reported_seroprevalence,
        covariates=covariates,
    )
    
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
            'age_pattern_root': age_pattern_root,
            'testing_root': testing_root,
            'day_inflection_list': day_inflection_list,
            'covariates': covariates,
            'covariate_list': covariate_list,
            'cross_variant_immunity': cross_variant_immunity,
            'verbose': verbose,
        }
        for n, (covariate_list, seroprevalence, sensitivity_data, cross_variant_immunity)
        in enumerate(zip(selected_combinations, seroprevalence_samples, sensitivity_data_samples, cross_variant_immunity_samples))
    }
    
    if verbose:
        logger.info('Storing inputs and submitting sero-sample jobs.')
    inputs_path = out_dir / 'pipeline_inputs.pkl'
    with inputs_path.open('wb') as file:
        pickle.dump(inputs, file, -1)
    pipeline_dir = out_dir / 'pipeline_outputs'
    shell_tools.mkdir(pipeline_dir)
    job_args_map = {n: [__file__, n, inputs_path, pipeline_dir] for n in range(n_samples)}
    cluster.run_cluster_jobs('covid_rates_pipeline', pipeline_dir, job_args_map)
    
    pipeline_results = {}
    for n in range(n_samples):
        with (pipeline_dir / str(n) / 'outputs.pkl').open('rb') as file:
            outputs = pickle.load(file)
        pipeline_results.update(outputs)
    
    em_data = estimates.excess_mortailty_scalars(model_inputs_root, excess_mortality)
    
    return pipeline_results, reported_seroprevalence, reported_sensitivity_data, \
           escape_variant_prevalence, severity_variant_prevalence, \
           vaccine_coverage, em_data


def pipeline(orig_seroprevalence: pd.DataFrame,
             shared: Dict,
             model_inputs_root: Path, excess_mortality: bool,
             sensitivity_data: pd.DataFrame,
             vaccine_coverage: pd.DataFrame,
             escape_variant_prevalence: pd.Series,
             severity_variant_prevalence: pd.Series,
             age_pattern_root: Path, testing_root: Path,
             day_inflection_list: List[str],
             covariates: List[pd.Series],
             covariate_list: List[str],
             cross_variant_immunity: float,
             storage_dir: Path, root_dir: Path,
             verbose: bool,) -> Tuple:
    if verbose:
        logger.info('\n*************************************\n'
                    f"IFR ESTIMATION -- testing inflection points: {', '.join(day_inflection_list)}\n"
                    '*************************************')
    ifr_input_data = ifr.data.load_input_data(model_inputs_root, excess_mortality, age_pattern_root,
                                              shared, orig_seroprevalence, sensitivity_data, vaccine_coverage,
                                              escape_variant_prevalence,
                                              severity_variant_prevalence,
                                              covariates,
                                              cross_variant_immunity,
                                              verbose=verbose)
    ifr_input_data_path = storage_dir / f'ifr_input_data.pkl'
    with ifr_input_data_path.open('wb') as file:
        pickle.dump(ifr_input_data, file, -1)
    covariate_list_path = storage_dir / f'covariate_list.pkl'
    with covariate_list_path.open('wb') as file:
        pickle.dump(covariate_list, file, -1)

    job_args_map = {}
    outputs_paths = []
    for day_inflection in day_inflection_list:
        outputs_path = storage_dir / f'{day_inflection}_outputs.pkl'
        outputs_paths.append(outputs_path)
        
        job_args_map.update({day_inflection: [ifr.runner.__file__, day_inflection,
                                              ifr_input_data_path, covariate_list_path, outputs_path]})
    
    cluster.run_cluster_jobs('covid_ifr_model', root_dir, job_args_map)

    full_ifr_results = {}
    for outputs_path in outputs_paths:
        with outputs_path.open('rb') as file:
            outputs = pickle.load(file)
        full_ifr_results.update(outputs)
    
    if verbose:
        logger.info('\n*************************************\n'
                    'IFR ESTIMATION -- determining best models and compiling adjusted seroprevalence\n'
                    '*************************************')
    ifr_nrmse, best_ifr_models, ifr_results, adj_seroprevalence, sensitivity, \
    cumul_reinfection_inflation_factor, daily_reinfection_inflation_factor = extract_ifr_results(full_ifr_results)

    if verbose:
        logger.info('\n*************************************\n'
                    'IDR ESTIMATION\n'
                    '*************************************')
    idr_input_data = idr.data.load_input_data(model_inputs_root, excess_mortality, testing_root,
                                              shared, adj_seroprevalence.copy(), vaccine_coverage.copy(),
                                              verbose=verbose)
    idr_results = idr.runner.runner(idr_input_data,
                                    shared,
                                    ifr_results.pred.copy(),
                                    daily_reinfection_inflation_factor.copy(),
                                    verbose=verbose)
    
    if verbose:
        logger.info('\n*************************************\n'
                    'IHR ESTIMATION\n'
                    '*************************************')
    ihr_input_data = ihr.data.load_input_data(model_inputs_root, age_pattern_root,
                                              shared, adj_seroprevalence.copy(), vaccine_coverage.copy(),
                                              escape_variant_prevalence.copy(),
                                              severity_variant_prevalence.copy(),
                                              covariates,
                                              verbose=verbose)
    ihr_results = ihr.runner.runner(ihr_input_data, daily_reinfection_inflation_factor.copy(),
                                    covariate_list,
                                    verbose=verbose)
    
    pipeline_results = {
        'covariate_list': covariate_list,
        'seroprevalence': adj_seroprevalence,
        'sensitivity': sensitivity,
        'cumul_reinfection_inflation_factor': cumul_reinfection_inflation_factor,
        'daily_reinfection_inflation_factor': daily_reinfection_inflation_factor,
        'day_inflection_list': day_inflection_list,
        'ifr_nrmse': ifr_nrmse,
        'best_ifr_models': best_ifr_models,
        'ifr_results': ifr_results,
        'idr_results': idr_results,
        'ihr_results': ihr_results,
    }
    
    return pipeline_results


def extract_ifr_results(full_ifr_results: Dict) -> Tuple:
    nrmse = []
    for day_inflection, di_results in full_ifr_results.items():
        nrmse.append(pd.concat([di_results['nrmse'],
                                pd.DataFrame({'day_inflection':day_inflection},
                                             index=di_results['nrmse'].index)],
                               axis=1))
    nrmse = pd.concat(nrmse).reset_index()
    best_models = (nrmse
                   .groupby('location_id')
                   .apply(lambda x: x.sort_values('nrmse')['day_inflection'].values[0])
                   .rename('day_inflection')
                   .reset_index())

    sensitivity = pd.DataFrame({'location_id':[]})
    cumul_reinfection_inflation_factor =[]
    daily_reinfection_inflation_factor =[]
    seroprevalence = []
    model_data = []
    mr_model_dict = {}
    pred_location_map = {}
    daily_numerator = []
    pred = []
    pred_unadj = []
    pred_fe = []
    pred_lr = []
    pred_hr = []
    pct_inf_lr = []
    pct_inf_hr = []
    age_stand_scaling_factor = []
    for location_id, day_inflection in zip(best_models['location_id'], best_models['day_inflection']):
        if location_id == 1:
            level_lambdas = full_ifr_results[day_inflection]['refit_results'].level_lambdas
        loc_seroprevalence = full_ifr_results[day_inflection]['refit_results'].seroprevalence
        loc_seroprevalence = loc_seroprevalence.loc[loc_seroprevalence['location_id'] == location_id]
        seroprevalence.append(loc_seroprevalence)
        
        loc_crif = full_ifr_results[day_inflection]['cumul_reinfection_inflation_factor']
        loc_crif = loc_crif.loc[loc_crif['location_id'] == location_id]
        cumul_reinfection_inflation_factor.append(loc_crif)
        
        loc_drif = full_ifr_results[day_inflection]['daily_reinfection_inflation_factor']
        loc_drif = loc_drif.loc[loc_drif['location_id'] == location_id]
        daily_reinfection_inflation_factor.append(loc_drif)

        loc_model_data = full_ifr_results[day_inflection]['refit_results'].model_data
        loc_model_data = loc_model_data.loc[loc_model_data['location_id'] == location_id]
        model_data.append(loc_model_data)
        
        try:  # extract pred map and model object in this chunk
            loc_model_location = full_ifr_results[day_inflection]['refit_results'].pred_location_map[location_id]
            pred_location_map.update({location_id: loc_model_location})
            if loc_model_location not in list(mr_model_dict.keys()):
                loc_mr_model = full_ifr_results[day_inflection]['refit_results'].mr_model_dict
                loc_mr_model = loc_mr_model[loc_model_location]
                mr_model_dict.update({loc_model_location: loc_mr_model})
            if loc_model_location not in sensitivity['location_id'].to_list():
                loc_sensitivity = full_ifr_results[day_inflection]['sensitivity']
                loc_sensitivity = loc_sensitivity.loc[loc_sensitivity['location_id'] == loc_model_location]
                sensitivity = pd.concat([sensitivity, loc_sensitivity])
        except KeyError:
            pass
        
        try:
            loc_daily_numerator = full_ifr_results[day_inflection]['refit_results'].daily_numerator.loc[[location_id]]
            daily_numerator.append(loc_daily_numerator)
        except KeyError:
            pass
        
        try:
            loc_pred = full_ifr_results[day_inflection]['refit_results'].pred.loc[[location_id]]
            pred.append(loc_pred)
        except KeyError:
            pass

        try:
            loc_pred_unadj = full_ifr_results[day_inflection]['refit_results'].pred_unadj.loc[[location_id]]
            pred_unadj.append(loc_pred_unadj)
        except KeyError:
            pass
        
        try:
            loc_pred_fe = full_ifr_results[day_inflection]['refit_results'].pred_fe.loc[[location_id]]
            pred_fe.append(loc_pred_fe)
        except KeyError:
            pass

        try:
            loc_pred_lr = full_ifr_results[day_inflection]['refit_results'].pred_lr.loc[[location_id]]
            pred_lr.append(loc_pred_lr)
        except KeyError:
            pass

        try:
            loc_pred_hr = full_ifr_results[day_inflection]['refit_results'].pred_hr.loc[[location_id]]
            pred_hr.append(loc_pred_hr)
        except KeyError:
            pass
        
        try:
            loc_pct_inf_lr = full_ifr_results[day_inflection]['refit_results'].pct_inf_lr.loc[[location_id]]
            pct_inf_lr.append(loc_pct_inf_lr)
        except KeyError:
            pass
        
        try:
            loc_pct_inf_hr = full_ifr_results[day_inflection]['refit_results'].pct_inf_hr.loc[[location_id]]
            pct_inf_hr.append(loc_pct_inf_hr)
        except KeyError:
            pass
        
        try:
            loc_age_stand_scaling_factor = full_ifr_results[day_inflection]['refit_results'].age_stand_scaling_factor.loc[[location_id]]
            age_stand_scaling_factor.append(loc_age_stand_scaling_factor)
        except KeyError:
            pass
        
    sensitivity = sensitivity.reset_index(drop=True)
    cumul_reinfection_inflation_factor = pd.concat(cumul_reinfection_inflation_factor).reset_index(drop=True)
    daily_reinfection_inflation_factor = pd.concat(daily_reinfection_inflation_factor).reset_index(drop=True)
    ifr_results = ifr.runner.RESULTS(
        seroprevalence=pd.concat(seroprevalence).reset_index(drop=True),
        model_data=pd.concat(model_data).reset_index(drop=True),
        mr_model_dict=mr_model_dict,
        pred_location_map=pred_location_map,
        level_lambdas=level_lambdas,
        daily_numerator=pd.concat(daily_numerator),
        pred=pd.concat(pred),
        pred_unadj=pd.concat(pred_unadj),
        pred_fe=pd.concat(pred_fe),
        pred_lr=pd.concat(pred_lr),
        pred_hr=pd.concat(pred_hr),
        pct_inf_lr=pd.concat(pct_inf_lr),
        pct_inf_hr=pd.concat(pct_inf_hr),
        age_stand_scaling_factor=pd.concat(age_stand_scaling_factor),
    )
    seroprevalence = ifr_results.seroprevalence.copy()

    return nrmse, best_models, ifr_results, seroprevalence, sensitivity, \
           cumul_reinfection_inflation_factor, daily_reinfection_inflation_factor


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
    
    ## working dir
    root_dir = Path(pipeline_dir) / str(n)
    storage_dir = root_dir / 'intermediate'
    shell_tools.mkdir(root_dir)
    shell_tools.mkdir(storage_dir)
    
    np.random.seed(123 * (n + 1))
    pipeline_outputs = pipeline(storage_dir=storage_dir, root_dir=root_dir,
                                **inputs)
    
    with (root_dir / 'outputs.pkl').open('wb') as file:
        pickle.dump({n: pipeline_outputs}, file)
        
    ## wipe intermediate datasets
    shutil.rmtree(storage_dir)
    

if __name__ == '__main__':
    configure_logging_to_terminal(verbose=2)
    
    main(int(sys.argv[1]), sys.argv[2], sys.argv[3])
