from typing import Tuple

import pandas as pd

from covid_historical_model.durations.durations import EXPOSURE_TO_DEATH

SEVERE_DISEASE_INFLATION = 1.3


def variants_vaccines(rate_age_pattern: pd.Series,
                      denom_age_pattern: pd.Series,
                      age_spec_population: pd.Series,
                      numerator: pd.Series,
                      rate: pd.Series,
                      variant_prevalence: pd.Series,
                      vaccine_coverage: pd.DataFrame,
                      population: pd.Series,
                      escape_variant_rate_scalar: float = SEVERE_DISEASE_INFLATION,):
    variant_prevalence = variant_prevalence.reset_index()
    variant_prevalence['date'] += pd.Timedelta(days=EXPOSURE_TO_DEATH)
    variant_prevalence = variant_prevalence.set_index(['location_id', 'date']).loc[:, 'variant_prevalence']
    variant_prevalence = pd.concat([rate, variant_prevalence], axis=1)  # borrow axis
    variant_prevalence = variant_prevalence['variant_prevalence'].fillna(0)

    vaccine_coverage = vaccine_coverage.reset_index()
    vaccine_coverage['date'] += pd.Timedelta(days=EXPOSURE_TO_DEATH)
    vaccine_coverage = vaccine_coverage.set_index(['location_id', 'date'])
    vaccine_coverage = pd.concat([rate.rename('rate'), vaccine_coverage], axis=1)  # borrow axis
    del vaccine_coverage['rate']
    vaccine_coverage = vaccine_coverage.fillna(0)
    
    location_ids = rate.reset_index()['location_id'].unique().tolist()
    location_ids = [l for l in location_ids if l in numerator.reset_index()['location_id'].unique().tolist()]
    numerator = numerator.loc[location_ids]
    
    numerator += 1e-4
    numerator /= population
    denominator_a = (numerator / rate[numerator.index])
    denominator_v = (numerator / (rate[numerator.index] * escape_variant_rate_scalar))
    denominator_a *= (1 - variant_prevalence[denominator_a.index])
    denominator_v *= variant_prevalence[denominator_v.index]

    numerator_a = (rate[denominator_a.index] * denominator_a)
    numerator_v = (rate[denominator_v.index] * escape_variant_rate_scalar * denominator_v)
    
    numerator_lr_a, numerator_hr_a, denominator_lr_a, denominator_hr_a = adjust_by_variant_classification(
        numerator_a,
        denominator_a,
        rate_age_pattern,
        denom_age_pattern,
        age_spec_population,
        vaccine_coverage,
        population,
        variant_suffix='wildtype',
    )
    numerator_lr_v, numerator_hr_v, denominator_lr_v, denominator_hr_v = adjust_by_variant_classification(
        numerator_v,
        denominator_v,
        rate_age_pattern,
        denom_age_pattern,
        age_spec_population,
        vaccine_coverage,
        population,
        variant_suffix='variant',
    )
    numerator_lr = numerator_lr_a + numerator_lr_v
    denominator_lr = denominator_lr_a + denominator_lr_v
    numerator_hr = numerator_hr_a + numerator_hr_v
    denominator_hr = denominator_hr_a + denominator_hr_v
    
    rate = (numerator_lr + numerator_hr) / (denominator_lr + denominator_hr)
    rate_lr = numerator_lr / denominator_lr
    rate_hr = numerator_hr / denominator_hr
    
    return rate, rate_lr, rate_hr


def adjust_by_variant_classification(numerator: pd.Series,
                                     denominator: pd.Series,
                                     rate_age_pattern: pd.Series,
                                     denom_age_pattern: pd.Series,
                                     age_spec_population: pd.Series,
                                     vaccine_coverage: pd.DataFrame,
                                     population: pd.Series,
                                     variant_suffix: str,):
    lr_rate_rr, hr_rate_rr = age_standardization.get_risk_group_rr(
        rate_age_pattern.copy(),
        denom_age_pattern.copy(),
        age_spec_population.copy(),
    )
    rate_lr = (numerator / denominator).fillna(0) * lr_rate_rr
    rate_hr = (numerator / denominator).fillna(0) * hr_rate_rr

    lr_denom_rr, hr_denom_rr = age_standardization.get_risk_group_rr(
        denom_age_pattern.copy(),
        denom_age_pattern.copy()**0,
        age_spec_population.copy(),
    )
    denominator_lr = denominator * lr_denom_rr
    denominator_hr = denominator * hr_denom_rr

    numerator_lr = rate_lr * denominator_lr
    numerator_hr = rate_hr * denominator_hr

    population_lr, population_hr = age_standardization.get_risk_group_populations(age_spec_population)

    numerator_lr, denominator_lr = vaccine_adjustments(
        numerator_lr, denominator_lr,
        vaccine_coverage[f'cumulative_lr_effective_{variant_suffix}'] / population_lr,
        vaccine_coverage[f'cumulative_lr_effective_protected_{variant_suffix}'] / population_lr,
    )
    numerator_hr, denominator_hr = vaccine_adjustments(
        numerator_hr, denominator_hr,
        vaccine_coverage[f'cumulative_hr_effective_{variant_suffix}'] / population_hr,
        vaccine_coverage[f'cumulative_hr_effective_protected_{variant_suffix}'] / population_hr,
    )

    numerator_lr *= population_lr
    numerator_hr *= population_hr

    denominator_lr *= population_lr
    denominator_hr *= population_hr        

    return numerator_lr, numerator_hr, denominator_lr, denominator_hr


def vaccine_adjustments(numerator: pd.Series,
                        denominator: pd.Series,
                        effective: pd.Series,
                        protected: pd.Series,) -> Tuple[pd.Series, pd.Series]:    
    numerator *= (1 - (effective[numerator.index] + protected[numerator.index]))
    denominator *= (1 - effective[denominator.index])

    return numerator, denominator
