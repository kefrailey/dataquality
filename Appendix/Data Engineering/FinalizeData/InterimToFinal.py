import datetime as dt
import pandas as pd
import boto3
import time
import sys
import os
from contextlib import contextmanager
from io import StringIO
from collections import defaultdict
from botocore.config import Config
from stations.station import df_from_s3_csv, df_to_s3_csv

ACCESS_KEY_ID = ## redacted
SECRET_ACCESS_KEY = ## redacted
creds_dict = {
    'aws_access_key_id': ACCESS_KEY_ID,
    'aws_secret_access_key': SECRET_ACCESS_KEY,
    'region_name': "us-west-2",
    'config': Config(s3={"use_accelerate_endpoint": True})
}


from_bucket = "interim-data"
to_bucket = "us-formatted-data"


prefix = "JHU"
now = time.strftime("%Y-%m-%d-%H-%M")
log_file = f'/home/ec2-user/Notebooks/log_processing_{prefix}_{now}.csv'


def df_to_s3_csv_SIO(bucket, key, df, **kwargs):
    s3_client = boto3.client('s3', **kwargs)

    with StringIO() as csv_buffer:
        df.to_csv(csv_buffer)
        response = s3_client.put_object(Bucket=bucket, Key=key, Body=csv_buffer.getvalue())
        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")

    return status

def df2s3(bucket, key, df, **kwargs):
    try:
        status = df_to_s3_csv(bucket, key, df, **creds_dict)
    except:
        status = df_to_s3_csv_SIO(bucket, key, df, **creds_dict)
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


@log_time
def raw_files_in_s3():
    s3 = boto3.resource('s3', **creds_dict)
    my_bucket = s3.Bucket(from_bucket)
    return [my_bucket_object.key for my_bucket_object in my_bucket.objects.all()]
      

def parse_s3_key(key):

    key_split = key.split("/")

    if len(key_split) == 5:
        prefix, country, state, county, file_name = key_split

        col_name, release = file_name.split("__")

        return prefix, country, state, county, col_name, release

    return False


def get_us_keys(all_s3_keys):
    "Return only US s3_keys in a dct {prefix/country/state/county/col_name : [s3_key]}"

    dct = defaultdict(list)

    for s3_key in all_s3_keys:

        vals = parse_s3_key(s3_key)

        if vals:

            prf, country, __, __, col_name, release = vals

            if country == 'US' and prf == prefix:

                new_key_prefix = s3_key.replace(f'__{release}', "")
                dct[new_key_prefix].append(s3_key)

    return dct

all_s3_keys = raw_files_in_s3()

@log_time
def finalize_data():

    clear_log()
    grouped_keys_dct = get_us_keys(all_s3_keys)

    for key_prefix, key_list in grouped_keys_dct.items():

        dfs = []

        for key in key_list:
            
            with log_stdout():
                df = df_from_s3_csv(from_bucket, key, **creds_dict)

            dfs.append(df)

        df = pd.concat(dfs, axis=1)
        key = f'{key_prefix}/data.csv'

        df2s3(to_bucket, key, df, **creds_dict)
        log(f'{key} completed')

    send_log_to_s3()

    return True


finalize_data()
