#!/usr/bin/env python
# coding: utf-8

import datetime as dt
import pandas as pd
import boto3
import time
import sys
import os
from contextlib import contextmanager
from io import StringIO
from collections import defaultdict
from stations.station import df_from_s3_csv, df_to_s3_csv



ACCESS_KEY_ID = # redacted
SECRET_ACCESS_KEY = # redacted
creds_dict = {
    'aws_access_key_id': ACCESS_KEY_ID,
    'aws_secret_access_key': SECRET_ACCESS_KEY
}


global_dir = 'csse_covid_19_data/csse_covid_19_daily_reports'
global_header = ['Country_Region', 'Province_State']

us_dir = 'csse_covid_19_data/csse_covid_19_daily_reports_us'
us_header = ['Country_Region', 'Province_State', 'Admin2']

short_root_path = '/home/ec2-user/'

from_bucket = "git-raw-data"
to_bucket = "interim-data"


prefix = "JHU"
repo = "COVID-19"
now = time.strftime("%Y-%m-%d-%H-%M")
to_process_file = f'{prefix}_to_process.txt'
log_file = f'log_processing_{prefix}_{now}.csv'
local_processing_dir = '/home/ec2-user/Processing'


def df_to_s3_csv_SIO(bucket, key, df, **kwargs):
    s3_client = boto3.client('s3', **kwargs)

    with StringIO() as csv_buffer:
        df.to_csv(csv_buffer)
        response = s3_client.put_object(Bucket=bucket, Key=key, Body=csv_buffer.getvalue())
        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")

    return status

def df2s3(bucket, key, df, **kwargs):
    try:
        status = df_to_s3_csv_SIO(bucket, key, df, **creds_dict)
    except:
        status = df_to_s3_csv(bucket, key, df, **creds_dict)
    return status


def log(message, log_file=log_file):
    
    now = time.strftime("%Y-%m-%d-%H-%M")
    with open(log_file, 'a') as f:
        f.write(f' {now}, {message}')
        f.write('\n')
        
def log_time(func):
    
    def inner(*args, **kwargs):
        start = time.time()
        start_dt = time.strftime("%Y-%m-%d-%H-%M")
        log(f'{func.__name__} started at {start_dt}')
        output = func(*args, **kwargs)
        end = time.time()
        duration = end - start
        log(f'{func.__name__} took {duration}')
        return output
    
    return inner

def clear_log(log_file=log_file):
    with open(log_file, 'w') as f:
        f.write("timestamp, message\n")
                
def send_log_to_s3(release=""):
    now = time.strftime("%Y-%m-%d-%H-%M")
    df = pd.read_csv(log_file)
    s3_key = f'log_{prefix}_{release}_{now}.csv'
    status = df2s3(to_bucket, s3_key, df, **creds_dict)
    log(f'log to s3 {status}')
    return status


@contextmanager
def log_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout



def extract_files_from_line(line):
    
    line = line.strip().split(" ")
    
    return line[0], line[1:]

@log_time
def raw_files_in_s3():
    session = boto3.Session(**creds_dict)
    s3 = session.resource('s3')
    my_bucket = s3.Bucket(from_bucket)
    return [my_bucket_object.key for my_bucket_object in my_bucket.objects.all()]
      
def get_val(row, interchangeable_col_names):
    """
    Get the value of a column, given:
        the column name may change
        or
        the column may not always be present
    """
    
    col_names_i = iter(interchangeable_col_names)
    col_name = next(col_names_i)
    val = None
    
    for col_name in interchangeable_col_names:
    
        try:
            val = row[col_name]
            
        except KeyError:
            pass
        
        if val is not None:
            return val
    
    return ""
    

def process_line(row):
        
    return {
        "state": str(get_val(row, ['Province_State',"Province/State"])).replace(",", "").replace('\"', ""),
        "country": str(get_val(row, ['Country_Region', "Country/Region"])).replace(",", "").replace('\"', ""),
        "county": str(get_val(row, ['Admin2'])).replace(",", "").replace('\"', ""),
        "confirmed": get_val(row, ["Confirmed"]),
        "deaths": get_val(row, ["Deaths"]),
        "recovered": get_val(row, ["Recovered"]),
        "active": get_val(row, ["Active"])
    }
    
    

def make_global_dct():
    
    return defaultdict(  # country
                        lambda: defaultdict(  # state
                            lambda: defaultdict(   # county
                                lambda: defaultdict(  # col_name
                                    lambda: defaultdict(  # release
                                        dict  # observation : value
                                    )))))


def update_dct(
    global_dct,
    release, obsv,
    country, state, county,
    deaths, confirmed, recovered,
    active=None
):
    
    global_dct[country][state][county]["deaths"][release][obsv] = deaths       
    global_dct[country][state][county]["confirmed"][release][obsv] = confirmed
    global_dct[country][state][county]["recovered"][release][obsv] = recovered
    
    if active is not None:
        global_dct[country][state][county]["active"][release][obsv] = active 
        
    return global_dct

@log_time
def send_to_s3(global_dct, release):
    
    for country, country_dct in global_dct.items():
        for state, state_dct in country_dct.items():
            for county, county_dct in state_dct.items():
                for col_name, vals_dct in county_dct.items():
                    df = pd.DataFrame(vals_dct)
                    s3_key = f'{prefix}/{country}/{state}/{county}/{col_name}__{release}.csv'
                    df2s3(bucket=to_bucket, key=s3_key, df=df, **creds_dict)
                    
    return True

@log_time   
def process_file(s3_key, release, global_dct):
    
    obsv = s3_key.split("/")[-1].strip(".csv")
    obsv_dt = dt.datetime.strptime(obsv, "%m-%d-%Y")
    obsv_s = obsv_dt.strftime("%Y-%m-%d")
    change_format_date = dt.datetime.strptime('03-22-2020', "%m-%d-%Y")
    
    with log_stdout():
        df = df_from_s3_csv(bucket=from_bucket, key=s3_key, **creds_dict)
        df.reset_index(inplace=True)
    
    for ind, row in df.iterrows():
        
        dct = process_line(row)
            
        global_dct = update_dct(global_dct, release, obsv_s, **dct)
        
    return global_dct


files_in_s3 = raw_files_in_s3()


@log_time
def process_release(release, exp_files, global_dct):

    for exp_file in exp_files:

        for dir_name in [us_dir, global_dir]:

            s3_key = f'{prefix}{repo}/{release}/{dir_name}/{exp_file}'

            if s3_key in files_in_s3:

                global_dct = process_file(s3_key, release, global_dct)
                
    return global_dct


@log_time
def process_releases(to_process_file):
    
    clear_log()
    
    processed_files_approx = 0
    global_dct = make_global_dct()
    
    with open(to_process_file, 'r') as f:

        for line in f:
            
            release, exp_files = extract_files_from_line(line)
            global_dct = process_release(release, exp_files, global_dct)
            processed_files_approx += len(exp_files)
            log(f'{release}')
            
            if processed_files_approx >= 5000:
                
                send_to_s3(global_dct, release)
                
                global_dct = make_global_dct()                
                processed_files_approx = 0
                log(f'COMPLETED {release} in s3')
                send_log_to_s3(release)
                clear_log()
                

    send_to_s3(global_dct, release)
    log(f'COMPLETED {release} in s3')
    send_log_to_s3(release)
    clear_log()

    
    return global_dct


dct = process_releases(to_process_file)


