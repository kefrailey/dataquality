#!/usr/bin/env python
# coding: utf-8

import numpy as np
import notebooks.src.visualizingrestatements.visualizingrestatements as vs
import datetime as dt
import pandas as pd
from contextlib import contextmanager
from io import StringIO
from collections import defaultdict
from botocore.config import Config
import matplotlib.pyplot as plt
import matplotlib.ticker as mtk
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates
from math import ceil
import numpy.typing as npt


Vector = npt.ArrayLike
Array = npt.ArrayLike  # column number and row number represent the same day, so diagonal should be ignored
DictDfs = dict
DictArrays = dict


def accuracy(value: float, truth: float)-> float:
    if truth != 0 and truth is not np.nan:
        return (truth - value)/truth

    if truth == 0:
        if value == 0:
            return 0
        if value != 0:
            return np.sign(value) * np.inf

    return np.nan


def offset_accuracy(X: npt.ArrayLike, K: int = 45) -> Vector:
    vals = []

    for i in range(X.shape[0] - K):
        t_hat = X[i, i + K]
        first_reported = X[i, i+1]  ## column number and row number represent the same day
        
        if t_hat == 0 or np.isnan(first_reported) or np.isnan(t_hat):
            vals.append(np.nan)

        else:
            vals.append( accuracy(first_reported, t_hat) )
    
    return vals


def naive_accuracy(X: npt.ArrayLike, m: int) -> Vector:

    vals = []

    for i in range(X.shape[0] -1):
        t_hat = X[i, m]
        first_reported = X[i, i+1]

        if t_hat == 0 or np.isnan( first_reported) or np.isnan(t_hat):
            vals.append(np.nan)

        else:
            vals.append( accuracy( first_reported, t_hat ))

    return vals


def estimate_with_average( measurements: Vector, history: int = 0) -> Vector:
    """Return moving or cumulative (history=0) average of not-nan values"""
    
    m_exists = [ 1 if not exists else 0 for exists in np.isnan( measurements)]    
    m_values = [ val if exists else 0 for exists, val in zip( m_exists, measurements)]

    if history == 0:
        denom = [ sum( m_exists[ 0 : i + 1]) for i in range( len( m_exists) )]
        numer = [ sum( m_values[ 0 : i + 1]) for i in range( len( m_values) )]

    else:
        denom = [ sum( m_exists[ max(0, i - history) : i + 1]) for i in range( len( m_exists) )]
        numer = [ sum( m_values[ max(0, i - history) : i + 1]) for i in range( len( m_values) )]

    estimates = [n/d if d != 0 else np.nan for n, d in zip(numer, denom)]

    return estimates


#  np.cumsum( [1,1])


def performance_estimates( measurements: Vector, offset: int, **kwargs) -> Vector:

    diffs = [np.nan] * offset
    ests = estimate_with_average( measurements, **kwargs) # rolling average

    for i in range( len( ests) - offset):
        diffs.append( measurements[ i + offset] - ests[ i])  # we can only use up to (today - offset) to estimate today's accuracy
    
    return diffs

## return the actual not the difference 
def accuracy_estimates_all( measurements_df: list, releases: list, **kwargs) -> pd.DataFrame:
    
    dct = {}

    for offset, accuracies in measurements_df.iteritems():
        waiting = [np.nan] * int(offset)
        ests =  estimate_with_average( accuracies, **kwargs) # rolling average

        dct[offset] = waiting + ests[: len(accuracies) -int(offset)]

    return pd.DataFrame(dct, index= releases)



def diff_matrix( measurements: list, offsets: int, columns: list, **kwargs) -> pd.DataFrame:
    
    perf_ests = []

    for accuracy_dm, offset in zip( measurements, offsets):
        perf_ests.append( performance_estimates(accuracy_dm, offset, **kwargs))


    return pd.DataFrame(perf_ests, index=[f'{each}' for each in offsets], columns=columns).T


# 

def completeness_and_timeliness( release: npt.ArrayLike, release_num: int)-> float:

    j = release_num - 1
    empty = True
    
    while empty:

        if release[j] is not np.nan:
            empty= False
        
        else:
            j -= 1

            if j < 0:
                return 0

    return (1 - (( (release_num - 1) - j)/ release_num)**2)
    


def timeliness_and_accuracy(observation: npt.ArrayLike, obsv_num: int, offset: int, epsilon: float) -> int:
    
    obsvs = observation[obsv_num + 1 : obsv_num + offset ]
    truth_estimate = obsvs[-1] # last reported value
    
    if truth_estimate is np.nan:
        return np.nan

    j = offset
    equal = True

    while equal:
        equal = ( abs( accuracy( truth_estimate, observation[ obsv_num + j]) ) < epsilon )

        if equal:
            j -= 1

            if j == 1:
                return 1

    return j

        
def timeliness_and_accuracy_all(X: npt.ArrayLike, offset: int, epsilon: float) -> Vector:

    taa = []
    for i in range(X.shape[0] - offset):
        taa.append(timeliness_and_accuracy(X[i, :], i, offset, epsilon))

    return taa


def completeness(X: Array, i: int) -> float:
    return X[i, i+1] is not np.nan

def completeness_group(df_dict : dict, i: int) -> float:

    m = len(df_dict .keys())
    indicators = []

    for key, df in df_dict .items():
        indicators.append( completeness(df, i) )

    return sum(indicators) / m

def check_dfs_dict(df_dict: dict ) -> tuple:

    key_iter = iter(df_dict )
    key = next(key_iter)
    columns, inds = df_dict [key].columns, df_dict [key].index
    keys = [key]

    while (key := next(key_iter, None)) is not None:
        col, ind = df_dict [key].columns, df_dict [key].index
        
        if (col != columns).any():
            return False, key, keys, "columns"

        if (ind != inds).any():
            return False, key, keys, "indices"
    
    return True, (inds, columns)

def completeness_group_all(arrays_dict: dict , df_dict: dict ) -> pd.DataFrame:
    # returns completeness of RELEASES, ie release i --> does X[i, i+1] exist
    
    aligned, tup = check_dfs_dict(df_dict )
    if not aligned:
        return "MISTMATCHED"

    inds, columns = tup
    vals = []

    # completeness for i, i+1
    for i in range(len(columns)-1):
        # i is obsv, i+1 is release
        vals.append( completeness_group(arrays_dict , i) )
    
    return pd.DataFrame([vals], columns=columns[1:], index=["Completeness"]).T


def major_restatement(i: int, release_i: npt.ArrayLike, release_ip1: npt.ArrayLike, beta: float=.2, alpha: float = 0) -> dict:

    revisions = sum( abs(release_i[: (i-1)] - release_ip1[: i-1]) > alpha)
    # a release i+1 can change observations 0 to i-1 (last value reported in release i)   (which is i values)
    if i == 0:
         return { "major_restatement": 0>beta,  "percent": 0,  "number": 0}   
    return { "major_restatement": (revisions/( i )) > beta,  "percent": (revisions/( i )),  "number": revisions}

def major_restatements_all(X: npt.ArrayLike, release_names: npt.ArrayLike, **kwargs) -> dict:

    mrs = {}
    for i in range(1, X.shape[1] -1): # no release 0
        mrs[release_names[i+1]] = major_restatement(i , X[:, i], X[:, i +1 ], **kwargs)

    return mrs

def major_restatements_group_all(arrays_dict: dict , df_dict:dict ,  **kwargs) -> pd.DataFrame:
    
    aligned, tup = check_dfs_dict(df_dict )
    if not aligned:
        return "MISTMATCHED"

    inds, columns = tup
    vals = []

    for key, X in arrays_dict .items():
        
        dct = major_restatements_all(X, columns,  **kwargs)
        for release, sub_dict  in dct.items():
            sub_dict["region"] = key
            sub_dict["release"] = release

            vals.append(sub_dict )
    
    return pd.DataFrame(vals)

def consistency_mr_concurrence(arrays_dict:dict , df_dict:dict , **kwargs) -> dict:

    ## use np array iterate thorugh dates

    aligned, tup = check_dfs_dict(df_dict )
    if not aligned:
        return "MISTMATCHED"
    
    inds, columns = tup
    vals = {}
    m = len(arrays_dict .keys())

    for i in range(len(columns) - 1):
        
        mrs = {}
        for region, X in arrays_dict .items():

            dct = major_restatement(i , X[:, i], X[:, i +1], **kwargs)
            mrs[region] = dct["major_restatement"]

        total = sum(mrs.values())
       
        dct = {
            region: 1 - abs(val - (1/ (m-1))*(total-val) )
            for region, val in mrs.items()
        }
        dct['group'] = 2 * abs((1/m) * total  - .5)

        vals[columns[i+1]] =  dct

    return vals


# # Validity

def validity(release: npt.ArrayLike, i: int) -> bool:

    return release[i] >= release[i-1]


def validity_all(X: npt.ArrayLike, release_names: npt.ArrayLike) -> dict:

    vals = {}
    for i in range(1, X.shape[1]-1):
        rel = release_names[i]
        vals[rel] = validity(X[:,i], i-1)

    return vals

def validity_group_all(arrays_dict: dict , df_dict:dict ,  **kwargs) -> pd.DataFrame:

    aligned, tup = check_dfs_dict(df_dict )
    if not aligned:
        return "MISTMATCHED"

    inds, columns = tup
    vals = {}

    for key, X in arrays_dict.items():
        
        vals[key] = validity_all(X, df_dict[key].columns)

    return pd.DataFrame(vals)


