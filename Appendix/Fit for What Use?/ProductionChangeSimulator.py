#!/usr/bin/env python
# coding: utf-8

"""Production Change Release Simulator

Simulates the effects of a change in data production
on releases
"""


import pandas as pd
import matplotlib.pyplot as plt
from copy import copy
import numpy as np
from math import floor, ceil
from collections import defaultdict
import matplotlib.ticker as mtk
import notebooks.src.visualizingrestatements.visualizingrestatements as vs


num_days = 10

confirmed = [floor(abs(2 - each)*1.75 + 5) for each in range(num_days)]

probable = [ceil(each*1.25) for each in confirmed]

days = range(num_days)

date_change = floor(num_days / 2)


def nonretroactive_change(confirmed, probable, date_change):
    vals = [c + p for c, p in zip(confirmed[:date_change], probable[:date_change])] + confirmed[date_change:]
    
    return pd.DataFrame([ vals[:ind + 1] for ind in range(len(vals))]).T


def retroactive_change(confirmed, probable, date_change):
    orig_vals = [c + p for c, p in zip(confirmed[:date_change], probable[:date_change])]
    
    vals = [ orig_vals[:ind + 1] for ind in range(date_change)]         + [confirmed[:ind + 1] for ind in range(date_change, num_days)]
    
    return pd.DataFrame(vals).T


def add_plot_shift(releases_df, shift=.05):
    df = pd.DataFrame()
    
    for i, col_info in enumerate(releases_df.iteritems()):
        name, vals = col_info
        df[f'Release {name + 1}'] = vals + (i * shift)
    
    return df


def plot_multiple_releases(
    title,
    confirmed, probable, date_change,
    func = nonretroactive_change,
    kind="Non-Retroactive Change"
):
    label = f'{kind}' 
    title_a = ": ".join(['Change in Values Reported', title])
    
    both = [p + c for p, c in zip(probable, confirmed)]
    releases = func(confirmed=confirmed, probable=probable, date_change=date_change)
    max_val = releases.max().max() * 1.2
    shift = max_val / 100
    releases_p = add_plot_shift(releases, shift)

    reds = plt.cm.Reds(np.linspace(0, 1, len(confirmed)))
    greens = plt.cm.Greens(np.linspace(0, 1, len(confirmed)))
    
    plt.style.use('seaborn-ticks')
    plt.figure(figsize=(10,7))
    ax = plt.gca()
    ax.plot(confirmed, 'g--', linewidth=1, label='confirmed')
    ax.plot(both, 'r--', linewidth=1, label='probable + confirmed')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel('Cases', fontsize=15)
    ax.set_xlabel('Observation Date', fontsize=15)
    plt.title(title_a + " " + label, fontsize = 20)
    plt.ylim(top=max_val, bottom = 0)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tick_params(
        axis='x',          
        which='both',      
        bottom=False,      
        top=False,         
        labelbottom=True
    ) 
    
    i = 0
    
    for ind, release in releases_p.iteritems():
        first_color = reds
        
        if kind=="Retroactive Change" and i >= date_change:
            first_color = greens
        
        ax.plot(release[:date_change+1], '-', color=first_color[i], linewidth=1, label=ind )
        ax.plot(release[date_change:], '-', color=greens[i], linewidth=1, label=ind )
        i += 1
     
    ax.text(0.95, 0.13, 'release date',
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='black', fontsize=15)
    ax.text(0.95, 0.1, '- - confirmed',
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='green', fontsize=15)
    ax.text(0.95, 0.07, f'   usable data',
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='green', fontsize=15)
    ax.text(0.95, 0.04, '- - probable + confirmed',
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='red', fontsize=15)
    ax.text(0.95, 0.01, f'  obsolete data',
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='red', fontsize=15)
    ax.get_yaxis().set_major_formatter(mtk.FuncFormatter(lambda x, p: format(int(x), ',')))
    
    i = 0
    for ind, release in releases_p.iteritems():   
        size = 70
        if i > 8 :
            size = 120
        ax.scatter( i, release[i], marker=r"$ {} $".format(i + 1), color="black", s=70)
        i += 1

    plt.show()
#     plt.savefig(f'../latex/plots/example_{title}_{kind}.png')
#     plt.clf()


plot_multiple_releases("Daily Cases",
                confirmed, probable, date_change)
plot_multiple_releases("Daily Cases",
                confirmed, probable, date_change,
                retroactive_change, "Retroactive Change")


confirmed_c, probable_c = list(np.cumsum(confirmed)), list(np.cumsum(probable))
plot_multiple_releases("Cumulative Cases",
                confirmed_c, probable_c,  date_change,
                nonretroactive_change, "Non-Retroactive Change")
plot_multiple_releases("Cumulative Cases",
                confirmed_c, probable_c, date_change,
                retroactive_change, "Retroactive Change")


