#!/usr/bin/env python
# coding: utf-8

"""Observation Date Simulator

Simulates the effects of lab capacity and testing access changes
on data recorded by sample date versus diagnosis date.
"""


import notebooks.src.visualizingrestatements.visualizingrestatements as vs


import pandas as pd
import matplotlib.pyplot as plt
from copy import copy
import numpy as np
import matplotlib.ticker as mticker


num_days = 10

samples = [10]*num_days

days = range(num_days)

capacity = [7]*num_days

increase = [1,1,2,3,5,8,13,21,34,55]


def report_by_diagnosis_date(samples, days, capacity):
    
    releases = []
    release = []
    backlog = 0
    
    for day in days: # everyday
        
        backlog += samples[day] # add today's samples to the backlog
        diagnosed = min(capacity[day], backlog) # diagnose as many as possible
        backlog -= diagnosed # remove the diagnosed from the backlog
        
        release = copy(release)
        release.append(diagnosed) # add today's diagnosed 
        
        releases.append(release) # add today's release
    
    return pd.DataFrame(releases, columns=days, index=days).T


def report_by_sample_date(samples, days, capacity):
    
    backlog = []
    releases = []
    release = []
    
    for day in days: # everyday

        c = capacity[day] # the capacity resets
        backlog.append(samples[day]) #add today's samples to the backlog
        diagnosed = [0]*len(backlog)
        i = 0
            
        for i, b in enumerate(backlog): #for each day in the backlog
            if c <1: # if there's still capacity left
                break
            d = min(b, c) # diagnose as much as you can
            diagnosed[i] = d # record your diagnosis
            c -= d # reduce the remaining capacity
            
                    
        backlog = [bd - dd for bd, dd in zip(backlog, diagnosed)] # update the backlog
        release = copy(release)
        release.append(0) # extend the release to today
        release = [rd + dd for rd, dd in zip(release, diagnosed)] # add the newly diagnosed
        releases.append(release)
        
    return pd.DataFrame(releases, columns=days, index=days).T


def add_plot_shift(releases_df, shift=.05):
    df = pd.DataFrame()
    
    for i, col_info in enumerate(releases_df.iteritems()):
        name, vals = col_info
        df[f'Release {name}'] = vals + (i * shift)
    
    return df


def backlog(capacity, samples):
    num = len(capacity)
    vals = [0]*num
    remainder = 0
    
    for i in range(num): 
        vals[i] = samples[i] + remainder - capacity[i]
        remainder = vals[i]
        
    return vals
        


def plot_with_truth(
    title,
    samples, days, capacity,
    func = report_by_diagnosis_date,
    kind="Diagnosis Date"
):

    label = f'Reporting by {kind}' 
    
    releases = func(samples, days, capacity)
    releases_p = add_plot_shift(releases)

    colors = plt.cm.Reds(np.linspace(0, 1, len(samples)))
    
    plt.style.use('../Style/plot.mplstyle')
    fig, axs = plt.subplots(1, 2, figsize=(20,7), sharex=True, sharey=True, squeeze=False)

    axs[0][0].plot(samples, 'b:', linewidth=4, label='samples')
    axs[0][0].plot(capacity, 'c--', linewidth=2, label='capacity')
    axs[0][0].plot(backlog(capacity, samples), linestyle='dashdot' , linewidth=2, label='backlog')
    axs[0][0].spines['top'].set_visible(False)
    axs[0][0].spines['right'].set_visible(False)
    plt.xlabel('Observation Date', fontsize=15)
    plt.ylabel('Tests', fontsize=15)
    plt.suptitle(label + ": " + title, fontsize = 24)
    releases_p.plot(xlim = (0,10), ylim=(0, 20), cmap="Reds", ax=axs[0][1], fontsize=15)
    axs[0][0].yaxis.set_major_locator(mticker.MultipleLocator(2))
    axs[0][0].legend(loc='upper left', bbox_to_anchor=(.9, .95), fontsize=15)
    axs[0][1].legend(loc='upper left', bbox_to_anchor=(.9, .95), fontsize=15)
    axs[0][0].tick_params(axis='both',labelsize=20)
    axs[0][1].tick_params(axis='x',labelsize=20)
    plt.plot()
    vs.df_to_latex(
    releases.astype('Int64') ,
    f'example_{title}_{kind}')
    plt.savefig(f'../latex/plots/example_{title}_{kind}.png', bbox_inches="tight")
    plt.clf()


table = plot_with_truth("Over Capacity",
                samples, days, capacity)


plot_with_truth("Over Capacity",
                samples, days, capacity,
                report_by_sample_date, "Sample Date")


plot_with_truth("Capacity Increase",
                samples, days, increase)


plot_with_truth("Capacity Increase",
                samples, days, increase,
                report_by_sample_date, "Sample Date")


plot_with_truth("Testing Access Increase",
                increase, days, capacity)
plot_with_truth("Testing Access Increase",
                increase, days, capacity,
                report_by_sample_date, "Sample Date")


