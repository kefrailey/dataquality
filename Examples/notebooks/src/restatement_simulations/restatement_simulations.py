# coding: utf-8

"""Simulates Releases and Common Restatement Types

Simulates the effects of a change in data on releases
"""


import pandas as pd
import datetime as dt
from copy import copy
import numpy as np
from collections import defaultdict
from math import floor, ceil
import random
from types import SimpleNamespace
from notebooks.src.visualizingrestatements.visualizingrestatements import example_plots


random.seed(217)
path = '../latex/plots/simulations2/'


def simulate_data(num_days):
    
    confirmed = [floor(abs(2 - each)*1.75 + 5) for each in range(num_days)]
    observations = list(range(num_days))
    
    return {
        "num_days": num_days,
        "confirmed": confirmed,
        "probable": [ceil(each*1.25) for each in confirmed],
        "days": range(num_days),
        "date_change": floor(num_days / 2),
        "release_dates": [d + 1 for d in observations],
        "observation_dates": observations
    }


def adjust_releases(vals, **kwargs):
    
    df = pd.DataFrame(np.nan, columns=kwargs['release_dates'], index=kwargs['observation_dates'])
    vals.columns = [r + 1 for r in vals.columns]
    vals = vals.reindex(sorted(vals.columns), axis=1)
    vals = vals.reindex(sorted(vals.index), axis=0)
    df.update(vals)
    df.columns =  pd.date_range(start='2024-01-02', periods=len(df.columns)).to_pydatetime()
    df.index = pd.date_range(start='2024-01-01', periods=len(df.index)).to_pydatetime()
    return df


def adjust_reporting(func):
    def inner(**kwargs):
        df = func(**kwargs)
        return adjust_releases(df, **kwargs)
    return inner


@adjust_reporting
def perfect_data(confirmed, **kwargs):
    
    dct = defaultdict(lambda: defaultdict(list))
    max_day = len(confirmed)
    
    for obsv, end_val in enumerate(confirmed):
        
        for d in range(obsv, max_day):
            dct[obsv][d] = end_val
            
    return pd.DataFrame(dct).T


@adjust_reporting
def missing_data(confirmed, **kwargs):
    random.seed(217)
    dct = defaultdict(lambda: defaultdict(list))
    max_day = len(confirmed)
    
    for obsv, end_val in enumerate(confirmed):
        
        for d in range(obsv, max_day):
            roll = random.randint(1, 6)
            if roll > 1:
                dct[obsv][d] = end_val
            
    return pd.DataFrame(dct).T


@adjust_reporting
def late_data(confirmed, lag=3, **kwargs):
    
    dct = defaultdict(lambda: defaultdict(list))
    max_day = len(confirmed)
    
    i = 0
    for obsv, end_val in enumerate(confirmed):
        
        update_day = lag*((obsv + 2) // lag ) 
            
        for d in range(update_day, max_day):
            dct[obsv][d] = end_val
            
        i+=1
    return pd.DataFrame(dct).T


@adjust_reporting
def provisional_data(confirmed, lag=3, delayed=.20, **kwargs):
    
    dct = defaultdict(lambda: defaultdict(list))
    max_day = len(confirmed)
    
    for obsv, end_val in enumerate(confirmed):
        
        update_day = obsv + lag
        orig_val = floor(end_val*(1-delayed))
        
        for d in range(obsv, min(update_day, max_day)):
            dct[obsv][d] = orig_val
            
        for d in range(update_day, max_day):
            dct[obsv][d] = end_val
            
    return pd.DataFrame(dct).T


@adjust_reporting
def staggered_data(confirmed, lag=3, delayed=.20, **kwargs):
    
    random.seed(217)
    
    dct = defaultdict(lambda: defaultdict(list))
    max_day = len(confirmed)
    
    for obsv, end_val in enumerate(confirmed):
        
        release=obsv
        this_perc = random.uniform(0, 1)
        staggers = range(random.randint(0,5))
        
        for stagger in staggers:
            
            release = obsv + stagger
            
            if release < max_day:    
                this_perc = random.uniform(this_perc,1)
                dct[obsv][release] = int(this_perc*end_val)

        for d in range(min(release, max_day), max_day):
            dct[obsv][d] = end_val

    
    return pd.DataFrame(dct).T


@adjust_reporting
def noisy_data(confirmed, **kwargs):
    """
    Creates noisy data through introducing random error and random reporting updates
    
    Second value introduced at observation date + x  (if x1 = 0, initial value never reported)
    Final value introduced at observation date + x  (if x2 = 0, second value never reported)
        x1 ~ Uniform(0,5), x2 ~Uniform(0, 2)
    
    Initial value = true vale * y1
    Second value = true value * y2
     Random variables y1 and y2 drawn from a truncated normal distribution
        mu = 1, sigma = .25, truncation at .5 and 1.5
    """
    
    random.seed(217)
    dct = defaultdict(lambda: defaultdict(list))
    max_day = len(confirmed)
    
    for obsv, end_val in enumerate(confirmed):
        
        error_1 = max(min(random.gauss(1,.25),1.5), .5)  # 2-sided truncation
        update_1 = obsv + random.randint(0,5)
        error_2 = max(min(random.gauss(1,.25),1.5), .5) # 2-sided truncation
        update_2 = update_1 + random.randint(0,2)

        for d in range(obsv, min(update_1, max_day)):
            dct[obsv][d] = floor(end_val*error_1)

        for d in range(min(update_1, max_day), min(update_2, max_day)):
            dct[obsv][d] = ceil(end_val*error_2)
            
        for d in range(min(update_2, max_day), max_day):
            dct[obsv][d] = end_val
            
    return pd.DataFrame(dct).T


@adjust_reporting
def nonretroactive_change(confirmed, probable, date_change, **kwargs):
    
    vals = [c + p for c, p in zip(confirmed[:date_change], probable[:date_change])]         + confirmed[date_change:]
    
    return pd.DataFrame([ vals[:ind + 1] for ind in range(len(vals))]).T


@adjust_reporting
def retroactive_change(confirmed, probable, date_change, num_days, **kwargs):
    
    orig_vals = [c + p for c, p in zip(confirmed[:date_change], probable[:date_change])]
    
    vals = [ orig_vals[:ind + 1] for ind in range(date_change)]         + [confirmed[:ind + 1] for ind in range(date_change, num_days)]
    
    return pd.DataFrame(vals).T


def make_plots(num_days, lag=3, delayed=.20):
    
    dct = simulate_data(num_days)
    dct['lag'] = lag
    dct['delayed'] = delayed
    
    example_plots(
        func = perfect_data,
        data_title=f'No Restatements, {num_days} Releases',
        dct=dct
    )
    
    example_plots(
        func = missing_data,
        data_title=f'Missing Data, {num_days} Releases',
        dct=dct
    )
    
    example_plots(
        func = late_data,
        data_title=f'Late Data, {num_days} Releases',
        dct=dct
    )
        
    example_plots(
        func = provisional_data,
        data_title=f'Provisional Data, {num_days} Releases, {str(delayed).replace(".", "p")} Delayed {lag} Days',
        dct=dct
    )
    
    example_plots(
        func = staggered_data,
        data_title=f'Staggered Data, {num_days} Releases',
        dct=dct
    )
    
    example_plots(
        func = nonretroactive_change,
        data_title=f'Non-Retroactive Change, {num_days} Releases',
        dct=dct
    )
    example_plots(
        func = retroactive_change,
        data_title=f'Retroactive Change, {num_days} Releases',
        dct=dct
    )
    example_plots(
        func = noisy_data,
        data_title=f'Noisy Data, {num_days} Releases',
        dct=dct
    )


# In[ ]:


make_plots(10)
make_plots(50)


