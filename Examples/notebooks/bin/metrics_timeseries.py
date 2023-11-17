#!/usr/bin/env python
# coding: utf-8

import numpy.typing as npt
import numpy as np


test_array = np.array(
    [[0,1,1,1, 1],
     [0,0,3,3, 3],
     [0,0,0,25, 25],
     [0,0,0,0, 24],
     [0,0,0,0, 0]
     ]
    )

test_array_nulls = np.array(
    [[0,1,1, np.nan, 1],
     [0,0,3, np.nan, 3],
     [0,0,0,25, 25],
     [0,0,0,0, 24],
     [0,0,0,0, 0]
     ]
    )

test_array_ff = np.array(
    [[0,1,1, 3, 1],
     [0,0,0, 0, 3], # obsv 1 IS ff
     [0,0,0, 0, 0], # obsv 2 is NOT ff
     [0,0,0, 0, 24],
     [0,0,0, 0, 0]
     ]
    )

test_array_dec = np.array(
    [[0, 3, 3, 3, 3],
     [0, 0, -1, -1, -1], # decrease
     [0, 0, 0, 2, -1], # decrease # 2
     [0, 0, 0, 0, 2],
     [0, 0, 0, 0, 0]
     ]
    )


Vector = npt.ArrayLike
Array = npt.ArrayLike 


def we_sum(summands: Vector) -> float:
    summands_ = [s for s in summands if ~np.isnan(s)]

    if len(summands_) > 0:
        return sum( [ ( i + 1)*val for i, val in enumerate(summands_) ])
    else:
        return np.nan


def we_average(summands: Vector) -> float:

    summands_ = [s for s in summands if ~np.isnan(s)]
    i = len(summands_)

    if i == 0:

        return np.NaN
    
    return (2 / ( i *( i + 1)))*we_sum(summands_)


def max_index_true(vals: Vector) -> int:
    return max( [ i * int( val ) for i, val in enumerate(vals)])


# # Accuracy

def accuracy(truth: float, val: float) -> float:
    
    if truth == val:
        return 0
    
    else:
        return truth - val


def measure_accuracy(release: Vector, truth: Vector, release_num: int) -> float:
    """Directly measure accuracy of a release, with increased weight to more recent observations"""

    summands = [accuracy(t, r) for t, r in zip(truth[:release_num], release[:release_num])]

    return we_average(summands)


def estimate_accuracy(releases: Array, offset: int) -> float:

    num_releases = releases.shape[1] - 1
    summands = []

    for i in range(1, num_releases - offset):

        release = releases[: i, i]
        truth = releases[: i, i + offset]
        
        summands.append(measure_accuracy(release, truth, i))

    summands_ = [s for s in summands if ~np.isnan(s)]

    if len(summands_) == 0:
        return 0
    return  sum(summands_) / len(summands_)


def measure_accuracy_release_history(releases, offset):
    num_measurable = releases.shape[1] - offset
    measured = []

    for i in range(1, num_measurable):
        release = releases[:i, i]
        truth = releases[:i, i + offset]
        measured.append( measure_accuracy( release, truth, i))

    return we_average(measured) 


def assess_accuracy_release_history( releases, offset):

    num_measurable = releases.shape[1] - offset
    measured = []

    for i in range(1, num_measurable):
        release = releases[:i, i]
        truth = releases[:i, i + offset]
        measured.append( measure_accuracy( release, truth, i))

    measured_ = [m for m in measured if ~np.isnan(m)]

    if len(measured_) > 0:
        estimated_val =  sum( measured_ )/ len(measured_)
        estimated = [estimated_val] * offset
        measured += estimated

    return we_average(measured) 


# # Consistency

# ## Nonretroactive Change

def modified_z_score(vals: Vector, val_to_evaluate: float) -> float:

    median_val = np.mean(np.median(vals))
    diffs = np.abs( vals - median_val)
    median_diff = np.median( diffs )

    if median_diff == 0:  ##### this is not right
        return (val_to_evaluate - median_val)/(1.253314* np.mean(diffs))
    
    else:

        return 0.6745 * (val_to_evaluate - median_val)/ median_diff


def check_nrc(release: Array, obsv_to_evaluate: int, alpha: float = 3.5, window: int = 21, neg=False) -> bool:
    """Checks if obsv_to_evaluate is an nonretroactive change"""

    # no values should be below 0


    vals = release[obsv_to_evaluate - window: obsv_to_evaluate]
    val_to_evaluate = release[obsv_to_evaluate]
    if neg and (val_to_evaluate < 0):
        return True
    mz_score =  modified_z_score(vals, val_to_evaluate)
    
    return (abs(mz_score) > alpha ) 


def nrcs_in_release(release: Vector, release_num: int, alpha:float = 3.5, window:int = 21, neg=False)-> Array:
    """Return list of indices of the nonretroactive changes """

    # raise exception if window >= release_num ?

    nrcs = []

    if window < release_num:
        for i in range(window, release_num):
            nrcs.append( (i, check_nrc(release, i, alpha, window, neg) ) )
    
    return nrcs


def nrcs_in_releases(releases: Array, **kwargs) -> Array:

    nrcs = np.full(releases.shape, False)
    num_releases = releases.shape[1] - 1

    for i in range(1, releases.shape[1]):

        release_nrcs = nrcs_in_release(releases[:, i], i, **kwargs)

        for obsv, val in release_nrcs:
            nrcs[obsv, i] = val

    return nrcs


def shared_nrcs(releases: Array, beta: float =.8, **kwargs) -> list:

    num_releases = releases.shape[1] - 1 # 0 is not a release
    
    nrcs = nrcs_in_releases(releases, **kwargs)
    
    num_nrcs = np.sum(nrcs, axis=1).tolist()

    #last possible nrc cant be shared because it has only appeared in last release
    shared = [ (num_nrc_i / (num_releases - i)) > beta for i, num_nrc_i in enumerate(num_nrcs[:-1])] 

    return shared


def most_recent_shared_nrc(releases: Array, **kwargs):

    vals = shared_nrcs(releases, **kwargs)

    return max_index_true(vals)


def most_recent_nrc(release: Array, release_num:int, **kwargs) -> int:

    vals = nrcs_in_release(release, release_num, **kwargs)
    nrcs = [ i * int(val) for i, val in vals]

    return max(nrcs)


release = [1,2,3,5,6,7,1,3,5,1,1000099999999999999905,1,0]

most_recent_nrc(release, 12, window=2)



nrcs = nrcs_in_releases(test_array, window=2)


# ## Major Restatement

def major_restatement(release: Vector, release_num: int, prior_release: Vector, alpha : float = 4, beta : float =.2) -> bool:

    summands = 0

    for obsv in range( release_num - 1):

        new_val = release[obsv]
        old_val = prior_release[obsv]

        if new_val !=0:
            indicator = abs(accuracy(old_val, new_val)) > alpha
            summands += indicator
        
        else:
            summands += int(new_val != old_val)

    return summands > (beta * ( release_num - 1))


def most_recent_major_restatement(release: Vector, release_num: int, prior_release : Vector, previous_mr :int =0, **kwargs) -> int:

    if major_restatement(release, release_num, prior_release, **kwargs):
        return release_num
    
    else:
        return previous_mr


def measure_consistency_mr(release_num: int, most_recent_mr: int=0) -> float:
    return (release_num - most_recent_mr) / release_num


major_restatement(test_array[:,3], 3, test_array[:,2])


# # Completeness

def measure_release_completeness(release: Vector, release_num: int) -> float:

    summands = [ ~np.isnan(val) for val in release[: release_num]]

    return we_average(summands)


measure_release_completeness([1, np.nan, 3, 4, 0,0,0,0,0], 6)


def measure_release_history_completeness(releases, release_num):

    summands = [measure_release_completeness(releases[:, i], i) for i in range(1, release_num + 1)]

    return we_average(summands)


measure_release_history_completeness(test_array_nulls, 4)


# # Timeliness, Completeness, and Believabilty

def was_filled_forward(observation: Vector, obsv_num, window: int = 7):

    if observation[obsv_num + 1] == 0:
       is_zero = [ val == 0 for val in observation[obsv_num + 1 : obsv_num + window + 1]]
       return not all(is_zero)


def estimate_filled_forward(releases, release_num, window=7):

    summands = []

    for i in range(release_num - window + 1): # obsv i - 1 will not have had a chance to be rewritten

        obsv = releases[i, : release_num ]

        if obsv[i+1] == 0:
            summands.append(was_filled_forward(obsv, i, window ))

    if len(summands) == 0:
        return np.nan

    return sum(summands) / len(summands)


estimate_filled_forward(test_array_ff, test_array_ff.shape[1]- 1, 2)


# # Validity and Believability

def measure_decreasing(releases, release_num):
    """ Assums releases are full of net increase
    """

    num_decreases = 0

    for i in range(release_num):

        obsvs = releases[i, i+1 : release_num + 1]
        decreases = [obsv < 0 for obsv in obsvs]

        num_decreases += int(any(decreases))

    return num_decreases


measure_decreasing(test_array_dec, test_array_dec.shape[1] - 1)


# # Believability

def measure_weekliness(release, release_num, num_weeks=7, alpha = .33):

    first_obsv = max(0, release_num - ( 7 * num_weeks))

    totals = []
    num_per_weekday = []

    # get the totals of nonneg values per 7 days chunks
    for i in range( num_weeks):
        if i < release_num:
            observations = [ day for day in range( min( first_obsv + i*7, release_num),  min(first_obsv + (i+1)*7, release_num) )]

            totals += [sum([ max(0, v) for v in release[observations]])]*len(observations)

    # get the value for each observation
    # for i in range(7):
    #     observations = [ day for day in range(first_obsv, release_num) if day % 7 == i ]
    #     num_per_weekday.append(sum(release[observations]))

    # get the percent of nonneg vals per 7 day chunk (edge case: when nonneg sum==0, then perc will be 0)
    observations = [ day for day in range(first_obsv, release_num) ]
    vals = [ max(0, num ) / max( 1, total) for num, total in zip(release[observations], totals)]

    # get median perc of 7 day chunk
    avg_percents = []
    for i in range(7):
        avg_percents.append( np.mean(np.median([v for ind, v in enumerate(vals) if ind % 7 == i])) )

    violation = [a > alpha for a in avg_percents]

    return sum(violation)


