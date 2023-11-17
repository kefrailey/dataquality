#!/usr/bin/env python
# coding: utf-8

"""
Find the optimal release dates to split data processing across machines
Because each day's data accumumaltes as x_i = x_i-1 + i  = ( i + 1 ) i/2
Early releases contain much less data that later releases
Processing as such is much more efficient

Assumes data was gathered from 
git log master.. --oneline --all --graph --decorate --date=iso-local --pretty=format:'commit::%h---time::%cd'
    $(git reflog | awk '{print $1}') > ../Acquire/JHUCommitHistory.txt
"""


import datetime as dt
from collections import defaultdict


earliest_date = dt.datetime(2020, 1, 18)
number_of_dates = 915
latest_date = earliest_date + dt.timedelta(days=number_of_dates)


def data_volume(x):
    return .5 * (x +1) * x


def rolling_sum(x):
    return [sum(x[:i+1]) for i in range(len(x))]


def find_split(vol, starting, ending, counter = 1):
    """
    Recursively finds best days to divide data processing across machines
    Counter allows for a maximum number of recusions (limiting total machines)
    """
    keep_going = True
    split = starting+1
    
    while keep_going:
        
        r_sum = rolling_sum( vol[ starting : split])[-1]
        l_sum = rolling_sum( vol[ split : ending])
        l_sum = l_sum[-1]
        net =  l_sum - r_sum 

        if net <= 0:
            lst = [split]
            keep_going = False

        else:
            split += 1
        
        if split == ending:
            lst = []
            counter = 0
            keep_going = False

    counter -=1
    
    if counter > 0:
        lst_l = find_split(vol, starting, split, counter-1)
        lst_r = find_split(vol, split, ending, counter-1)
        lst = lst_l + lst + lst_r 
    return lst 


vol = [data_volume(x) for x in range(1, number_of_dates + 1)]


splits = find_split(vol, 0, len(vol)-1, 4)


dates_dt = [earliest_date] + [earliest_date + dt.timedelta(d) for d in splits] + [latest_date]


dates = [dt.datetime.strftime(d, "%Y-%m-%d") for d in dates_dt]


def divide_commits(dates, prefix="JHU"):
    
    dct = defaultdict(dict)

    with open(f'{prefix}CommitHistory.txt', 'r') as f:
        
        for line in f:
            line = line.strip()
            line = line.split("---")

            if len(line) > 1:
                line = [each.split("::")[1] for each in line]
                commit = line[0]
                date = line[1].split(" ")[0]
                time = line[1].split(" ")[1]
                dct[date][time] = commit
                commit, date, time
    
    to_write = defaultdict(list)
    pairs = [(s, e) for s, e in zip(dates[:-1], dates[1:])]
    
    pairs_i = iter(pairs)
    start, end = next(pairs_i)

    ordered_dates = sorted(dct.keys())
    for date in ordered_dates:

        if date >= end:
            try:
                start, end = next(pairs_i)
                lines = []
            
            except StopIteration:
                break
                
        line = [date]
        ordered_times = sorted(dct[date].keys(), reverse=True)

        for time in ordered_times:

            commit = dct[date][time]
            line.append(commit)
        
        to_write[start].append(" ".join(line))
        
    for div, lines in to_write.items():
        with open(f'{prefix}_{div}.txt', 'w') as f:
            for line in lines:
                f.write(line)
                f.write('\n')
        
    return to_write
        


def releases_per():
    
    releases = [earliest_date + dt.timedelta(days=days) for days in range(number_of_dates)]

    jhu_lines = []
    obsv_lines = []
    
    splits_i = iter(dates_dt)
    
    current_split = next(splits_i)
    next_split = next(splits_i)
    
    for release in releases:
        
        diff = release - earliest_date
        obsv_dates = [earliest_date + dt.timedelta(days=days) for days in range(diff.days)]
        
        observations = [dt.datetime.strftime(d, "%Y-%m-%d") for d in obsv_dates]
        obsv_line = [dt.datetime.strftime(release, "%Y-%m-%d")] + observations
        obsv_lines.append(" ".join(obsv_line))

        jhu_line = [dt.datetime.strftime(release, "%Y-%m-%d")] +             [dt.datetime.strftime(d, "%m-%d-%Y.csv") for d in obsv_dates] +             [dt.datetime.strftime(release, "%m-%d-%Y.csv")]
        jhu_lines.append((" ").join(jhu_line))

        if release == next_split:
            
            with open(dt.datetime.strftime(current_split, "JHU_files_%Y-%m-%d.txt"), 'w') as f:
                for line in jhu_lines:
                    f.write(line + '\n')
            current_split = next_split
            next_split = next(splits_i)
            jhu_lines = []
        
    with open(dt.datetime.strftime(release, "expected_observations.txt"), 'w') as f:
        for line in obsv_lines:
            f.write(line + '\n')
    return True


releases_per()


output = divide_commits(dates)
output = divide_commits(dates, "NYT")


