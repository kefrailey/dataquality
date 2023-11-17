#!/usr/bin/env python
# coding: utf-8

import subprocess
import pandas as pd
from random import randint
import boto3

from datetime import date, datetime, timedelta
from time import sleep
import sys
import os
import copy

import pathlib

import time


ACCESS_KEY_ID = ## redacted
SECRET_ACCESS_KEY = ## redacted

prefix = "JHU"

root_s = '/home/ec2-user'
repo_as = '/home/ec2-user/COVID-19'
repo_rs = 'COVID-19'
dirs_as = [
    '/home/ec2-user/COVID-19/csse_covid_19_data/csse_covid_19_daily_reports_us/', 
    '/home/ec2-user/COVID-19/csse_covid_19_data/csse_covid_19_daily_reports/'
]
dirs_rs = [
    '/csse_covid_19_data/csse_covid_19_daily_reports_us/', 
    '/csse_covid_19_data/csse_covid_19_daily_reports/'
]


commit_file = 'JHU_commits.txt'
now = time.strftime("%Y-%m-%d-%H-%M")
log_file = f'log_{prefix}_{now}.txt'


def log(message, log_file=log_file):
    with open(log_file, 'a') as f:
        f.write(message)
        f.write('\n')

def log_time(func):
    
    def inner(*args, **kwargs):
        start = time.time()
        start_dt = time.strftime("%Y-%m-%d-%H-%M")
        log(f'{func.__name__} started at {start_dt}')
        output = func(*args, **kwargs)
        end = time.time()
        duration = end - start
        end_dt = time.strftime("%Y-%m-%d-%H-%M")
        log(f'{func.__name__} started at {end_dt}')
        log(f'{func.__name__} took {duration}')
        return output
    
    return inner


def parse_line(line):
    """Returns date and commits from line of format 'date commit commit ....\n'"""
    line = line.strip()
    line = line.split(" ")
    return line[0], line[1:]


@log_time
def checkout(commit):
    
    stash = subprocess.run(
        [f'git -C {repo_as} stash'], stdout=subprocess.PIPE, shell=True)
    checkout_s = f'git -C {repo_as} checkout {commit}'
    checkout_result = subprocess.run(
        [checkout_s], stdout=subprocess.PIPE, shell=True)
    
    if checkout_result.returncode == 0:
        return True, checkout_result
    
    else:
        log(f'   {commit} {checkout_result.returncode}' )
        return False, checkout_result
    
@log_time
def clear_git_chages():
    clear_cache_s = f'git -C {repo_as} rm --cached -r .'
    clear_cachet_result = subprocess.run(
        [clear_cache_s], stdout=subprocess.PIPE, shell=True)
    reset_s = f'git -C {repo_as} reset --hard '
    reset_result = subprocess.run(
        [reset_s], stdout=subprocess.PIPE, shell=True)
    
    if reset_result.returncode == 0:
        return True, reset_result
    
    else:
        return False, reset_result


def check_directory_exists(dir_as):
    
    dir_p = pathlib.Path(dir_as)
    if dir_p.is_dir():
        return True
    else:
        return False
        
@log_time
def files_to_s3(dir_as, dir_rs, date):

    s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY_ID,aws_secret_access_key=SECRET_ACCESS_KEY)
    
    for file_rs in os.listdir(dir_as):

        file_as = f'{dir_as}/{file_rs}'
        file_ap = pathlib.Path(file_as)

        if file_ap.is_file():
            
            with open(file_as, "rb") as f:
                
                msg = s3.upload_fileobj(f, 'git-raw-data', f'{prefix}{repo_rs}/{date}{dir_rs}{file_rs}')
                log(f'{date} {dir_rs} {file_rs} {msg}')


def download_and_upload_files():
    
    start = time.time()
    cleared, msg = clear_git_chages()
    
    if not cleared:
        return msg

    with open(commit_file, 'r') as f:

        for line in f:
                
            dirs_as_copy = copy.copy(dirs_as)
            date, commits = parse_line(line)
            commits_i = iter(commits)
            looking = True
            log(f'\n{date} begun')
            
            while looking:

                try:
                    commit = next(commits_i)

                except StopIteration:
                    looking = False
                    log(f'{date} Not found')
                    break

                checked_out, checkout_msg = checkout(commit)
                
                if checked_out:
                    
                    for dir_as in dirs_as_copy:

                        if check_directory_exists(dir_as):
                            
                            dir_rs = dir_as.replace(repo_as,"")
                            log(f'{date} {commit} {dir_rs}')
                            files_to_s3(dir_as, dir_rs, date)

                            log(f'{date} {dir_rs} completed')
                            
                            dirs_as_copy.remove(dir_as)
                            
                    if len(dirs_as_copy) == 0:
                        looking = False


    end = time.time()
    duration = end - start
    log(f'Process took {duration}')

    return True
                


download_and_upload_files()


