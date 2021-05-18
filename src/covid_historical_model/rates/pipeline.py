from typing import List, Tuple, Dict
from pathlib import Path
from loguru import logger
import dill as pickle

import pandas as pd
import numpy as np

from covid_shared import shell_tools
from covid_shared.cli_tools.logging import configure_logging_to_terminal

from covid_historical_model.etl import model_inputs, estimates
from covid_historical_model.rates import serology
from covid_historical_model.rates import age_standardization
from covid_historical_model.rates import ifr
from covid_historical_model.rates import idr
from covid_historical_model.rates import ihr
from covid_historical_model import cluster
from covid_historical_model.rates import location_plots
from covid_historical_model.utils.pdf_merger import pdf_merger


def pipeline_wrapper(out_dir: Path,
                     model_inputs_root: Path, excess_mortality: bool,
                     vaccine_coverage_root: Path, variant_scaleup_root: Path,
                     age_pattern_root: Path, testing_root: Path,
                     day_inflection_list: List[str] = ['2020-05-01', '2020-06-01', '2020-07-01', '2020-08-01',
                                                       '2020-09-01', '2020-10-01', '2020-11-01', '2020-12-01',],
                     n_samples: int = 100,
                     verbose: bool = True,) -> Tuple:
    if verbose:
        logger.info('Loading variant, vaccine, and sero data.')
    escape_variant_prevalence = estimates.variant_scaleup(variant_scaleup_root, 'escape', verbose=verbose)
    severity_variant_prevalence = estimates.variant_scaleup(variant_scaleup_root, 'severity', verbose=verbose)
    vaccine_coverage = estimates.vaccine_coverage(vaccine_coverage_root)

    seroprevalence_samples = serology.load_seroprevalence_sub_vacccinated(
        model_inputs_root, vaccine_coverage.copy(), n_samples=n_samples,
        verbose=verbose,
    )
    
    if verbose:
        logger.info('Submitting sero-sample jobs.')
    pipeline_dir = out_dir / 'pipeline_outputs'
    shell_tools.mkdir(pipeline_dir)
    inputs = {
        n: 
              {
                  'seroprevalence': seroprevalence,
                  'pipeline_dir': pipeline_dir,
                  'model_inputs_root': model_inputs_root,
                  'excess_mortality': excess_mortality,
                  'vaccine_coverage_root': vaccine_coverage_root,
                  'variant_scaleup_root': variant_scaleup_root,
                  'age_pattern_root': age_pattern_root,
                  'testing_root': testing_root,
                  'day_inflection_list': day_inflection_list,
                  'verbose': verbose,
              }
        for n, seroprevalence in enumerate(seroprevalence_samples)
    }
    
    inputs_path = out_dir / 'pipeline_inputs.pkl'
    with inputs_path.open('wb') as file:
        pickle.dump(inputs, file, -1)
    job_args_map = {n: [__file__, n, inputs_path, outputs_path]}
    cluster.run_cluster_jobs('covid_rates_pipeline', out_dir, job_args_map)
    
    em_data = estimates.excess_mortailty_scalars(model_inputs_root, excess_mortality)
    
    # stuff for plotting
    hierarchy = model_inputs.hierarchy(model_inputs_root)
    population = model_inputs.population(model_inputs_root)
    age_spec_population = model_inputs.population(model_inputs_root, by_age=True)
    population_lr, population_hr = age_standardization.get_risk_group_populations(age_spec_population)
    sensitivity_data = model_inputs.assay_sensitivity(model_inputs_root)
    del age_spec_population
    
    ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
    logger.warning('NEED THINK ABOUT RIGHT APPROACH TO PLOTTING.')
    ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
    
    return None


def pipeline(seroprevalence: pd.DataFrame,
             model_inputs_root: Path, excess_mortality: bool,
             vaccine_coverage_root: Path, variant_scaleup_root: Path,
             age_pattern_root: Path, testing_root: Path,
             day_inflection_list: List[str],
             verbose: bool,) -> Tuple:
    if verbose:
        logger.info('\n*************************************\n'
                    f"IFR ESTIMATION -- testing inflection points: {', '.join(day_inflection_list)}\n"
                    '*************************************')
    ifr_input_data = ifr.data.load_input_data(model_inputs_root, excess_mortality, age_pattern_root,
                                              seroprevalence, vaccine_coverage,
                                              escape_variant_prevalence,
                                              severity_variant_prevalence,
                                              verbose=verbose)
    job_args_map = {}
    outputs_paths = []
    for day_inflection in day_inflection_list:
        inputs = {
            'input_data': ifr_input_data, 'day_inflection': day_inflection
        }
        # ifr.runner.runner(**inputs)
        # raise ValueError('STOP')
        
        inputs_path = storage_dir / f'{day_inflection}_inputs.pkl'
        with inputs_path.open('wb') as file:
            pickle.dump(inputs, file, -1)
        
        outputs_path = storage_dir / f'{day_inflection}_outputs.pkl'
        outputs_paths.append(outputs_path)
        
        job_args_map.update({day_inflection: [ifr.runner.__file__, inputs_path, outputs_path]})
    
    cluster.run_cluster_jobs('covid_ifr_model', storage_dir, job_args_map)

    full_ifr_results = {}
    for outputs_path in outputs_paths:
        with outputs_path.open('rb') as file:
            outputs = pickle.load(file)
        full_ifr_results.update(outputs)
    
    if verbose:
        logger.info('\n*************************************\n'
                    'IFR ESTIMATION -- determining best models and compiling adjusted seroprevalence\n'
                    '*************************************')
    ifr_nrmse, best_ifr_models, ifr_results, adj_seroprevalence, \
    sensitivity, reinfection_inflation_factor = extract_ifr_results(full_ifr_results)

    if verbose:
        logger.info('\n*************************************\n'
                    'IDR ESTIMATION\n'
                    '*************************************')
    idr_input_data = idr.data.load_input_data(model_inputs_root, excess_mortality, testing_root,
                                              seroprevalence, vaccine_coverage, verbose=verbose)
    idr_results = idr.runner.runner(idr_input_data,
                                    ifr_results.pred.copy(),
                                    reinfection_inflation_factor.copy(),
                                    verbose=verbose)
    
    if verbose:
        logger.info('\n*************************************\n'
                    'IHR ESTIMATION\n'
                    '*************************************')
    ihr_input_data = ihr.data.load_input_data(model_inputs_root, age_pattern_root,
                                              adj_seroprevalence.copy(), vaccine_coverage.copy(),
                                              escape_variant_prevalence.copy(),
                                              severity_variant_prevalence.copy(),
                                              verbose=verbose)
    ihr_results = ihr.runner.runner(ihr_input_data, reinfection_inflation_factor.copy(),
                                    verbose=verbose)
    
    pipeline_results = {
        'seroprevalence':seroprevalence,
        'reinfection_inflation_factor':reinfection_inflation_factor,
        'ifr_nrmse':ifr_nrmse,
        'best_ifr_models':best_ifr_models,
        'ifr_results':ifr_results,
        'idr_results':idr_results,
        'ihr_results':ihr_results,
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
    reinfection_inflation_factor =[]
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
        
        loc_rif = full_ifr_results[day_inflection]['reinfection_inflation_factor']
        loc_rif = loc_rif.loc[loc_rif['location_id'] == location_id]
        reinfection_inflation_factor.append(loc_rif)

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
    reinfection_inflation_factor = pd.concat(reinfection_inflation_factor).reset_index(drop=True)
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

    return nrmse, best_models, ifr_results, seroprevalence, sensitivity, reinfection_inflation_factor


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
    pdf_merger(pdf_paths, pdf_location_names, pdf_parent_names, pdf_levels, str(pdf_out_path))

    
def submit_plots():
    plot_inputs = {
        'best_ifr_models': best_ifr_models.set_index('location_id'),
        'seroprevalence': seroprevalence,
        'sensitivity': sensitivity,
        'sensitivity_data': sensitivity_data,
        'reinfection_inflation_factor': reinfection_inflation_factor,
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


def main(n: int, inputs_path: str):
    with Path(inputs_path).open('rb') as file:
        inputs = pickle.load(file)[n]
    
    ## working dir
    storage_dir = inputs['pipeline_dir'] / str(n) / 'intermediate'
    results_dir = inputs['pipeline_dir'] / str(n) / 'results'
    # plots_dir = inputs['pipeline_dir'] / str(n) / 'plots'
    shell_tools.mkdir(storage_dir)
    shell_tools.mkdir(results_dir)
    # shell_tools.mkdir(plots_dir)
    
    pipeline_outputs = pipeline(**inputs)
    
    with (inputs['pipeline_dir'] / 'outputs.pkl').open('wb') as file:
        pickle.dump({n: pipeline_outputs}, file)
    

if __name__ == '__main__':
    configure_logging_to_terminal(verbose=2)
    
    main(int(sys.argv[1]), sys.argv[2])
