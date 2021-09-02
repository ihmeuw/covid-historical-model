from pathlib import Path
from typing import Dict
from collections import namedtuple
from datetime import datetime

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import seaborn as sns

from covid_historical_model.durations.durations import EXPOSURE_TO_DEATH

DATE_LOCATOR = mdates.AutoDateLocator(maxticks=10)
DATE_FORMATTER = mdates.ConciseDateFormatter(DATE_LOCATOR, show_offset=False)

START_DATE = pd.Timestamp('2020-03-01')
END_DATE = pd.Timestamp(str(datetime.today().date()))

L_RAW = 'lightgrey'
D_RAW = 'darkgrey'

L_REFIT = 'royalblue'
D_REFIT = 'mediumblue'

RESID_0 = 'black'

PCT_INF = 'firebrick'

VACC_A = 'seagreen'
VACC_V = 'darkorange'

VAR_E = 'darkorchid'
VAR_S = 'gold'

##

LEGEND = 8
YLAB = 12
TITLE = 24


def plotter(location_id: int, location_name: str,
            out_path: Path,
            full_ifr_results: Dict, best_ifr_models: pd.Series, ifr_results: namedtuple,
            population: pd.Series, population_lr: pd.Series, population_hr: pd.Series,
            vaccine_coverage: pd.DataFrame,
            escape_variant_prevalence: pd.Series,
            severity_variant_prevalence: pd.Series,
            **kwargs,):
    n_dates = len(full_ifr_results)
    n_cols = n_dates * 3
    n_rows = 6
    widths = [1] * n_cols
    heights = [2, 2, 3, 3, 3, 3]

    sns.set_style('whitegrid')
    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    gs = fig.add_gridspec(n_rows, n_cols, width_ratios=widths, height_ratios=heights)
    
   # get residual axis limits
    residuals = []
    for day_inflection, day_ifr_results in full_ifr_results.items():
        residuals.append(day_ifr_results['residuals'].loc[location_id].reset_index())
    residuals = pd.concat(residuals)
    min_residual = residuals['residuals'].min()
    max_residual = residuals['residuals'].max()
    min_residual -= (max_residual - min_residual) * .1
    max_residual += (max_residual - min_residual) * .1
    del residuals
    
    modeled_location = location_id in ifr_results.mr_model_dict.keys()
    filler_date_index = pd.Index([START_DATE], name='date')

    for i, (day_inflection, day_ifr_results) in enumerate(full_ifr_results.items()):
        residuals = day_ifr_results['residuals'].loc[location_id].reset_index()  # first index is pred_location
        if modeled_location:
            model_data_location_ids = (day_ifr_results['refit_results']
                                       .mr_model_dict[location_id]
                                       .data.to_df()['study_id'].unique().tolist())
        else:
            model_data_location_ids = []

        nrmse = day_ifr_results['nrmse'].loc[location_id]
        
        raw_model_data = day_ifr_results['raw_results'].model_data
        raw_model_data = raw_model_data.set_index(['location_id']).loc[model_data_location_ids]
        
        raw_pred = day_ifr_results['raw_results'].pred_unadj
        try:
            raw_pred = raw_pred.loc[location_id]
        except KeyError:
            raw_pred = pd.Series(np.nan, index=filler_date_index)
            
        refit_model_data = day_ifr_results['refit_results'].model_data
        refit_model_data = refit_model_data.set_index(['location_id']).loc[model_data_location_ids]
        
        refit_pred = day_ifr_results['refit_results'].pred_unadj
        try:
            refit_pred = refit_pred.loc[location_id]
        except KeyError:
            refit_pred = pd.Series(np.nan, index=filler_date_index)

        if day_inflection == best_ifr_models.loc[location_id].item():
            plot_title = f'{day_inflection}**\n{np.round(nrmse, 6)}'
        else:
            plot_title = f'{day_inflection}\n{np.round(nrmse, 6)}'

        fit_ax = fig.add_subplot(gs[0, i*3:i*3 + 3])    
        fit_plot(fit_ax, i, raw_model_data, raw_pred, refit_model_data, refit_pred, plot_title)

        resid_ax = fig.add_subplot(gs[1, i*3:i*3 + 3])
        resid_plot(resid_ax, i, residuals, min_residual, max_residual)

    for i, group in enumerate(['All age', '<65', '65+']):
        if group == 'All age':
            pred_ifr = ifr_results.pred
            pct_inf = ifr_results.pct_inf_lr + ifr_results.pct_inf_hr
            e_a = (vaccine_coverage['cumulative_hr_effective_wildtype'] + vaccine_coverage['cumulative_lr_effective_wildtype']) / population
            ep_a = e_a + (vaccine_coverage['cumulative_hr_effective_protected_wildtype'] + vaccine_coverage['cumulative_lr_effective_protected_wildtype']) / population
            e_v = (vaccine_coverage['cumulative_hr_effective_variant'] + vaccine_coverage['cumulative_lr_effective_variant']) / population
            ep_v = e_v + (vaccine_coverage['cumulative_hr_effective_protected_variant'] + vaccine_coverage['cumulative_lr_effective_protected_variant']) / population
            e_a += e_v
            ep_a += ep_v
        elif group == '<65':
            pred_ifr = ifr_results.pred_lr
            pct_inf = ifr_results.pct_inf_lr
            e_a = vaccine_coverage['cumulative_lr_effective_wildtype'] / population_lr
            ep_a = e_a + vaccine_coverage['cumulative_lr_effective_protected_wildtype'] / population_lr
            e_v = vaccine_coverage['cumulative_lr_effective_variant'] / population_lr
            ep_v = e_v + vaccine_coverage['cumulative_lr_effective_protected_variant'] / population_lr
            e_a += e_v
            ep_a += ep_v
        elif group == '65+':
            pred_ifr = ifr_results.pred_hr
            pct_inf = ifr_results.pct_inf_hr
            e_a = vaccine_coverage['cumulative_hr_effective_wildtype'] / population_hr
            ep_a = e_a + vaccine_coverage['cumulative_hr_effective_protected_wildtype'] / population_hr
            e_v = vaccine_coverage['cumulative_hr_effective_variant'] / population_hr
            ep_v = e_v + vaccine_coverage['cumulative_hr_effective_protected_variant'] / population_hr
            e_a += e_v
            ep_a += ep_v
        else:
            raise ValueError('Invalid group')

        try:
            pred_ifr = pred_ifr.loc[location_id]
        except KeyError:
            pred_ifr = pd.Series(np.nan, index=filler_date_index)

        try:
            pct_inf = pct_inf.loc[location_id]
        except KeyError:
            pct_inf = pd.Series(np.nan, index=filler_date_index)

        try:
            e_a = e_a.loc[location_id]
            e_a.index += pd.Timedelta(days=EXPOSURE_TO_DEATH)
            ep_a = ep_a.loc[location_id]
            ep_a.index += pd.Timedelta(days=EXPOSURE_TO_DEATH)
            e_v = e_v.loc[location_id]
            e_v.index += pd.Timedelta(days=EXPOSURE_TO_DEATH)
            ep_v = ep_v.loc[location_id]
            ep_v.index += pd.Timedelta(days=EXPOSURE_TO_DEATH)
        except KeyError:
            e_a = pd.Series(np.nan, index=filler_date_index)
            ep_a = pd.Series(np.nan, index=filler_date_index)
            e_v = pd.Series(np.nan, index=filler_date_index)
            ep_v = pd.Series(np.nan, index=filler_date_index)

        try:
            escape = escape_variant_prevalence.loc[location_id]
            escape.index += pd.Timedelta(days=EXPOSURE_TO_DEATH)
            severity = severity_variant_prevalence.loc[location_id]
            severity.index += pd.Timedelta(days=EXPOSURE_TO_DEATH)
        except KeyError:
            escape = pd.Series(np.nan, index=filler_date_index)
            severity = pd.Series(np.nan, index=filler_date_index)

        ifr_ax = fig.add_subplot(gs[2, i*n_dates:i*n_dates + n_dates])
        ifr_plot(ifr_ax, i, pred_ifr, group)

        pct_inf_ax = fig.add_subplot(gs[3, i*n_dates:i*n_dates + n_dates])
        pct_inf_plot(pct_inf_ax, i, pct_inf)

        vaccine_ax = fig.add_subplot(gs[4, i*n_dates:i*n_dates + n_dates])
        vaccine_plot(vaccine_ax, i, e_a, ep_a, e_v, ep_v)

        variant_ax = fig.add_subplot(gs[5, i*n_dates:i*n_dates + n_dates])
        variant_plot(variant_ax, i, escape, severity)

    fig.suptitle(f'{location_name} ({location_id})', fontsize=TITLE)
    if out_path is None:
        fig.show()
    else:
        fig.savefig(out_path, bbox_inches='tight')
        plt.close(fig)

    
def fit_plot(ax, i, raw_model_data, raw_pred, refit_model_data, refit_pred, plot_title):
    ax.scatter(raw_model_data['mean_death_date'],
               raw_model_data['ifr'] * 100, marker='^', alpha=0.5,
               c=L_RAW, edgecolors=D_RAW)
    ax.plot(raw_pred * 100, color=D_RAW)
    ax.scatter(refit_model_data['mean_death_date'],
               refit_model_data['ifr'] * 100, marker='o', alpha=0.5,
               c=L_REFIT, edgecolors=D_REFIT)
    ax.plot(refit_pred * 100, color=D_REFIT)
    if i == 0:
        ax.set_ylabel('IFR (%)', fontsize=YLAB)
    ax.set_title(plot_title)
    ax.set_xlim(START_DATE, END_DATE)
    ax.set_xticklabels([])


def resid_plot(ax, i, residuals, y_min, y_max):
    ax.scatter(residuals['date'], residuals['residuals'],
               color=L_REFIT, edgecolors=D_REFIT, alpha=0.5)
    ax.axhline(0, linestyle='--', color=RESID_0)
    ax.set_ylim(y_min, y_max)
    if i == 0:
        ax.set_ylabel('Seroprevalence\nresiduals', fontsize=YLAB)
    ax.set_xlim(START_DATE, END_DATE)
    ax.xaxis.set_major_locator(DATE_LOCATOR)
    ax.xaxis.set_major_formatter(DATE_FORMATTER)
    ax.tick_params('x', labelrotation=60)
    
    
def ifr_plot(ax, i, pred_ifr, plot_title):
    ax.plot(pred_ifr * 100, color=D_REFIT)
    if i == 0:
        ax.set_ylabel('IFR (%)', fontsize=YLAB)
    ax.set_title(plot_title)
    ax.set_xlim(START_DATE, END_DATE)
    ax.set_xticklabels([])
    
    
def pct_inf_plot(ax, i, pct_inf):
    ax.plot(pct_inf, color=PCT_INF)
    if i == 0:
        ax.set_ylabel('Infection fraction', fontsize=YLAB)
    ax.set_xlim(START_DATE, END_DATE)
    # ax.set_ylim(0, 100)
    ax.set_xticklabels([])
    
    
def vaccine_plot(ax, i, e_a, ep_a, e_v, ep_v):
    ax.plot(ep_a * 100, color=VACC_A, label='Effective + protected (ancestral + variant)')
    ax.plot(e_a * 100, color=VACC_A, linestyle='--', label='Effective (ancestral + variant)')
    ax.plot(ep_v * 100, color=VACC_V, label='Effective + protected (ancestral)')
    ax.plot(e_v * 100, color=VACC_V, linestyle='--', label='Effective (ancestral)')
    if i == 0:
        ax.set_ylabel('Vaccinated (%)', fontsize=YLAB)
        ax.legend(loc=2, fontsize=LEGEND)
    ax.set_xlim(START_DATE, END_DATE)
    ax.set_ylim(0, 100)
    ax.set_xticklabels([])
    

def variant_plot(ax, i, escape, severity):
    ax.plot(escape, color=VAR_E, label='P1 + B.1.351 + B.1.617')
    ax.plot(severity, color=VAR_S, label='B.1.1.7')
    if i == 0:
        ax.set_ylabel('Variant prevalence', fontsize=YLAB)
        ax.legend(loc=2, fontsize=LEGEND)
    ax.set_xlim(START_DATE, END_DATE)
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_locator(DATE_LOCATOR)
    ax.xaxis.set_major_formatter(DATE_FORMATTER)
