from typing import Tuple, List, Dict
import multiprocessing
from tqdm import tqdm

import pandas as pd

from covid_historical_model.rates import age_standardization
from covid_historical_model.cluster import OMP_NUM_THREADS

SEVERE_DISEASE_INFLATION = 1.29


def variants_vaccines(rate_age_pattern: pd.Series,
                      denom_age_pattern: pd.Series,
                      age_spec_population: pd.Series,
                      rate: pd.Series,
                      day_shift: int,
                      escape_variant_prevalence: pd.Series,
                      severity_variant_prevalence: pd.Series,
                      vaccine_coverage: pd.DataFrame,
                      population: pd.Series,):
    escape_variant_prevalence = escape_variant_prevalence.reset_index()
    escape_variant_prevalence['date'] += pd.Timedelta(days=day_shift)
    escape_variant_prevalence = (escape_variant_prevalence
                                 .set_index(['location_id', 'date'])
                                 .loc[:, 'escape_variant_prevalence'])
    escape_variant_prevalence = pd.concat([rate, escape_variant_prevalence], axis=1)  # borrow axis
    escape_variant_prevalence = escape_variant_prevalence['escape_variant_prevalence'].fillna(0)
    
    severity_variant_prevalence = severity_variant_prevalence.reset_index()
    severity_variant_prevalence['date'] += pd.Timedelta(days=day_shift)
    severity_variant_prevalence = (severity_variant_prevalence
                                 .set_index(['location_id', 'date'])
                                 .loc[:, 'severity_variant_prevalence'])
    severity_variant_prevalence = pd.concat([rate, severity_variant_prevalence], axis=1)  # borrow axis
    severity_variant_prevalence = severity_variant_prevalence['severity_variant_prevalence'].fillna(0)

    vaccine_coverage = vaccine_coverage.reset_index()
    vaccine_coverage['date'] += pd.Timedelta(days=day_shift)
    vaccine_coverage = vaccine_coverage.set_index(['location_id', 'date'])
    vaccine_coverage = pd.concat([rate.rename('rate'), vaccine_coverage], axis=1)  # borrow axis
    del vaccine_coverage['rate']
    vaccine_coverage = vaccine_coverage.fillna(0)
    
    # not super necessary...
    numerator = pd.Series(100, index=rate.index)
    numerator /= population
    
    denominator_a = (numerator / rate)
    denominator_ev = (numerator / (rate * SEVERE_DISEASE_INFLATION))
    denominator_sv = (numerator / (rate * SEVERE_DISEASE_INFLATION))
    denominator_a *= (1 - (escape_variant_prevalence + severity_variant_prevalence)[denominator_a.index])
    denominator_ev *= escape_variant_prevalence[denominator_ev.index]
    denominator_sv *= severity_variant_prevalence[denominator_sv.index]

    numerator_a = (rate * denominator_a)
    numerator_ev = (rate * SEVERE_DISEASE_INFLATION * denominator_ev)
    numerator_sv = (rate * SEVERE_DISEASE_INFLATION * denominator_sv)
    
    shared_args = {
        'rate_age_pattern': rate_age_pattern,
        'denom_age_pattern': denom_age_pattern,
        'age_spec_population': age_spec_population,
        'vaccine_coverage': vaccine_coverage,
        'population': population,
    }
    args_a = {
        'numerator': numerator_a,
        'denominator': denominator_a,
        'variant_suffixes': ['wildtype', 'variant',],
    }
    args_a.update(shared_args)
    args_sv = {
        'numerator': numerator_sv,
        'denominator': denominator_sv,
        'variant_suffixes': ['wildtype', 'variant'],
    }
    args_sv.update(shared_args)
    args_ev = {
        'numerator': numerator_ev,
        'denominator': denominator_ev,
        'variant_suffixes': ['variant',],
    }
    args_ev.update(shared_args)
    
    args = [args_a, args_sv, args_ev]
    with multiprocessing.Pool(int(OMP_NUM_THREADS)) as p:
        adj_results = list(tqdm(p.imap(adjust_by_variant_classification, args), total=len(args), file=sys.stdout))
    numerator_lr_a, numerator_hr_a, denominator_lr_a, denominator_hr_a = adj_results[0]
    numerator_lr_sv, numerator_hr_sv, denominator_lr_sv, denominator_hr_sv = adj_results[1]
    numerator_lr_ev, numerator_hr_ev, denominator_lr_ev, denominator_hr_ev = adj_results[2]
    del args, adj_results
    
    numerator_lr = numerator_lr_a + numerator_lr_ev + numerator_lr_sv
    denominator_lr = denominator_lr_a + denominator_lr_ev + denominator_lr_sv
    numerator_hr = numerator_hr_a + numerator_hr_ev + numerator_hr_sv
    denominator_hr = denominator_hr_a + denominator_hr_ev + denominator_hr_sv
    
    rate = (numerator_lr + numerator_hr) / (denominator_lr + denominator_hr)
    rate_lr = numerator_lr / denominator_lr
    rate_hr = numerator_hr / denominator_hr
    
    pct_inf_lr = denominator_lr / (denominator_lr + denominator_hr)
    pct_inf_hr = denominator_hr / (denominator_lr + denominator_hr)
    
    return rate, rate_lr, rate_hr, pct_inf_lr, pct_inf_hr


def adjust_by_variant_classification(kwargs: Dict,):
    # manually assign these, multiprocessing didn't like trad kwargs
    numerator: pd.Series = kwargs['numerator']
    denominator: pd.Series = kwargs['denominator']
    rate_age_pattern: pd.Series = kwargs['rate_age_pattern']
    denom_age_pattern: pd.Series = kwargs['denom_age_pattern']
    age_spec_population: pd.Series = kwargs['age_spec_population']
    vaccine_coverage: pd.DataFrame = kwargs['vaccine_coverage']
    population: pd.Series = kwargs['population']
    variant_suffixes: List[str] = kwargs['variant_suffixes']
    
    lr_rate_rr, hr_rate_rr = age_standardization.get_risk_group_rr(
        rate_age_pattern.copy(),
        denom_age_pattern.copy()**0,  # REMOVE THIS IF WE WANT TO USE THE ACTUAL SERO AGE PATTERN
        age_spec_population.copy(),
    )
    rate_lr = (numerator / denominator) * lr_rate_rr
    rate_hr = (numerator / denominator) * hr_rate_rr

    lr_denom_rr, hr_denom_rr = age_standardization.get_risk_group_rr(
        denom_age_pattern.copy()**0,  # REMOVE THIS IF WE WANT TO USE THE ACTUAL SERO AGE PATTERN
        denom_age_pattern.copy()**0,  # REMOVE THIS IF WE WANT TO USE THE ACTUAL SERO AGE PATTERN
        age_spec_population.copy(),
    )
    denominator_lr = denominator * lr_denom_rr
    denominator_hr = denominator * hr_denom_rr

    numerator_lr = rate_lr * denominator_lr
    numerator_hr = rate_hr * denominator_hr

    population_lr, population_hr = age_standardization.get_risk_group_populations(age_spec_population)

    lr_e = [f'cumulative_lr_effective_{variant_suffix}' for variant_suffix in variant_suffixes]
    lr_ep = [f'cumulative_lr_effective_protected_{variant_suffix}' for variant_suffix in variant_suffixes]
    numerator_lr, denominator_lr = vaccine_adjustments(
        numerator_lr, denominator_lr,
        vaccine_coverage[lr_e].sum(axis=1) / population_lr,
        vaccine_coverage[lr_ep].sum(axis=1) / population_lr,
    )
    hr_e = [f'cumulative_hr_effective_{variant_suffix}' for variant_suffix in variant_suffixes]
    hr_ep = [f'cumulative_hr_effective_protected_{variant_suffix}' for variant_suffix in variant_suffixes]
    numerator_hr, denominator_hr = vaccine_adjustments(
        numerator_hr, denominator_hr,
        vaccine_coverage[hr_e].sum(axis=1) / population_hr,
        vaccine_coverage[hr_ep].sum(axis=1) / population_hr,
    )

    numerator_lr *= population_lr
    numerator_lr = numerator_lr.fillna(0)
    numerator_hr *= population_hr
    numerator_hr = numerator_hr.fillna(0)

    denominator_lr *= population_lr
    denominator_lr = denominator_lr.fillna(0)
    denominator_hr *= population_hr
    denominator_hr = denominator_hr.fillna(0)

    return numerator_lr, numerator_hr, denominator_lr, denominator_hr


def vaccine_adjustments(numerator: pd.Series,
                        denominator: pd.Series,
                        effective: pd.Series,
                        protected: pd.Series,) -> Tuple[pd.Series, pd.Series]:    
    numerator *= (1 - (effective[numerator.index] + protected[numerator.index]))
    denominator *= (1 - effective[denominator.index])

    return numerator, denominator
