# coding: utf-8

# %matplotlib inline


path = '../latex/plots/CA_deaths_JHU/'


import notebooks.src.visualizingrestatements.visualizingrestatements as vs


import datetime as dt
import numpy as np
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
import matplotlib.pyplot as plt
import matplotlib.ticker as mtk
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates
from math import ceil


ACCESS_KEY_ID = 'AKIA456IG3QPJIO72JV4'
SECRET_ACCESS_KEY = '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'
creds_dict = {
    'aws_access_key_id': ACCESS_KEY_ID,
    'aws_secret_access_key': SECRET_ACCESS_KEY,
    'region_name': "us-west-2",
    'config': Config(s3={"use_accelerate_endpoint": True})
}


state = "California"
col = "deaths"
data_title = f'CA {col.title()}'
df = df_from_s3_csv(bucket='us-formatted-data', key=f'JHU/US/{state}//{col}/data.csv', **creds_dict)
df.index = pd.to_datetime(df.index, format="%Y-%m-%d")
df.columns = pd.to_datetime(df.columns, format="%Y-%m-%d")


rdict = vs.major_restatements(df, 0)


spring2020_df = df.loc["2020-03-01":"2020-04-30","2020-03-01":"2020-04-30"]
spring2020_title = "March-April 2020 CA Deaths"


summer2021_df = df.loc["2021-05-01":"2021-06-30","2021-05-01":"2021-06-30"]
summer2021_title = "May-June 2021 CA Deaths"
isummer2021_df = df.loc["2021-07-01":"2021-08-30","2021-07-01":"2021-08-30"]
isummer2021_title = "July-Aug 2021 CA Deaths"
fall2021_df = df.loc["2021-10-01":"2021-11-30","2021-10-01":"2021-11-30"]
fall2021_title = "Oct-Nov 2021 CA Deaths"
summer2022_df = df.loc["2022-04-01":"2022-05-30","2022-04-01":"2022-05-30"]
summer2022_title = "April-May 2022 CA Deaths"


# # Stacked Line

# ## No Stacking

fig, ax, dct = vs.plot_multiple_releases(spring2020_df,data_title=spring2020_title, shift=0, alpha=1.0, show=True)


fig.savefig(f'{path}spring2020_line.png',  bbox_inches = "tight")


fig, ax, dct = vs.plot_multiple_releases(summer2021_df,data_title=summer2021_title, shift=0, alpha=1, show=True)


fig.savefig(f'{path}summer2021_line.png', bbox_inches = "tight")


# ## Stacked

fig, ax, dct = vs.plot_multiple_releases(isummer2021_df,data_title=isummer2021_title, shift=15, alpha=.75, show=True)


ax.text(.67 , .42, f'NR Change', color='grey', fontsize=15, transform=ax.transAxes)
ax.text(.52 , .3, f'Surge\nor NR Change', color='grey', fontsize=15, transform=ax.transAxes)
fig


fig.savefig(f'{path}isummer2021_stacked_line.png', bbox_inches = "tight")


fig, ax, dct = vs.plot_multiple_releases(spring2020_df,data_title=spring2020_title, show=True, alpha=1)


x = spring2020_df.columns[25]
y = 1300
ax.text(x , y, f'Missing Data', color='grey', fontsize=20) #, transform=ax.transAxes)


fig


fig.savefig(f'{path}spring2020_stacked_line.png', bbox_inches = "tight")


fig, ax, dct = vs.plot_multiple_releases(summer2021_df,data_title=summer2021_title, shift=12, alpha=.75, show=True)


fig.savefig(f'{path}summer2021_stacked_line.png', bbox_inches = "tight")


# # Matrix

# ## Heatmap

fig, __, __ = vs.plot_staircase(df ,data_title=data_title, show=True, heatmap=True)
fig.savefig(f'{path}all_heatmap.png', bbox_inches = "tight")


fig, ax, dct = vs.plot_staircase(summer2021_df ,data_title=summer2021_title, show=True, heatmap=True)


fig.savefig(f'{path}summer2021_heatmap.png', bbox_inches = "tight")


# ## Between Releases

fig, ax, dct = vs.plot_between_release_changes(df ,data_title="CA Deaths", missing=False, show=True)
fig.savefig(f'{path}between_release.png', bbox_inches = "tight")


fig, ax, dct = vs.plot_between_release_changes(summer2021_df ,data_title=summer2021_title, show=True)


ax.text(.35 , .75, f'Prior to\nRetroactive\nChange', color='grey', fontsize=20, transform=ax.transAxes)
ax.text(.75 , .75, f'After\nRetroactive\nChange', color='grey', fontsize=20, transform=ax.transAxes)


fig


fig.savefig(f'{path}summer2021_between_release_annotated.png', bbox_inches = "tight")


# ## Within Releases

fig, ax, dct = vs.plot_within_release_changes(
    df ,data_title=data_title + " (Decreases)",
    missing=False, show=True,
    alpha = 1, window=1, func=lambda x: 0, positive=False, percent=False)


fig.savefig(f'{path}within_release_negative.png', bbox_inches = "tight")


# In[ ]:


within_release_changes_dct = {"alpha": 2, "window": 14, "func": "avg"}
fig, ax, dct = vs.plot_within_release_changes(
    df, data_title=data_title + " (Major Changes)",
    missing=False, show=True,
    **within_release_changes_dct)
fig.savefig(f'{path}within_release.png', bbox_inches = "tight")


# In[ ]:


fig, ax, dct = vs.plot_staircase(isummer2021_df ,data_title=isummer2021_title, show=True, heatmap=True)


# In[ ]:


ax.text(.345 , .305, f'Suspected Decrease->', color='grey', fontsize=15, transform=ax.transAxes)


# In[ ]:


fig


# In[ ]:


fig.savefig(f'{path}isummer2021_heatmap_annotated.png', bbox_inches = "tight")


# In[ ]:


fig, ax, dct = vs.plot_within_release_changes(
    isummer2021_df ,data_title=isummer2021_title,
    show=True,
    **within_release_changes_dct)


# In[ ]:


ax.text(.345 , .305, f'Confirmed Decrease->', color='grey', fontsize=15, transform=ax.transAxes)
fig


# In[ ]:


fig.savefig(f'{path}isummer2021_within_release.png', bbox_inches = "tight")


# In[ ]:


ax.text(.6 , .465, f'Surge or NR Change', color='grey', fontsize=20, transform=ax.transAxes)
ax.text(.7 , .34, f'NR Change', color='grey', fontsize=20, transform=ax.transAxes)


# In[ ]:


fig


# In[ ]:


fig.savefig(f'{path}isummer2021_within_release_annotated.png', bbox_inches = "tight")


# # Restatement Lags

# In[ ]:


fig, ax, dct = vs.plot_restatement_lags(df ,data_title, show=True)


# In[ ]:


fig.savefig(f'{path}lags.png', bbox_inches = "tight")


# In[ ]:


fig, ax, dct = vs.plot_restatement_lags(isummer2021_df ,data_title=isummer2021_title, show=True)
ax.text(.5 , .5, f'No indication\nof Non-Retroactive Change', color='grey', fontsize=20, transform=ax.transAxes, ha='center')


# In[ ]:


fig


# In[ ]:


fig.savefig(f'{path}isummer2021_lags.png', bbox_inches = "tight")


# In[ ]:


fig, ax, dct = vs.plot_restatement_lags(summer2021_df ,data_title=summer2021_title, show=True)


# In[ ]:


ax.text(.25 , .4, f'Retroactive Change', color='grey', fontsize=20, transform=ax.transAxes, rotation=-37)
ax.text(.07 , .06,
        f'Indicates Initial Values Were Provisional',
        color='grey', fontsize=13, transform=ax.transAxes)


# In[ ]:


fig


# In[ ]:


fig.savefig(f'{path}summer2021_lags.png', bbox_inches = "tight")


# ## Bifrost Plots

# In[ ]:


fig, ax, dct = vs.plot_bifrost(df ,data_title, show=True)


# In[ ]:


ax.text(.15 , .025, f'Adjustment Made Late 2021-Early 2022', color='blue', fontsize=12, transform=ax.transAxes)
ax.text(.2 , .875, f'Adjustment Made\nSummer 2022', color='magenta', fontsize=12, transform=ax.transAxes)
fig


# In[ ]:


fig.savefig(f'{path}bifrost.png', bbox_inches = "tight")


# In[ ]:


fig, ax, dct = vs.plot_bifrost(summer2021_df ,summer2021_title + "", show=True)


# In[ ]:


fig.savefig(f'{path}summer2021_bifrost.png', bbox_inches = "tight")


# In[ ]:


fig, ax, dct = vs.plot_bifrost(summer2021_df.diff() ,summer2021_title + " (Daily)", show=True, y_scale="symlog")


# In[ ]:


fig.savefig(f'{path}summer2021_bifrost_daily.png', bbox_inches = "tight")


# In[ ]:


fig, ax, dct = vs.plot_bifrost(summer2021_df.diff().rolling(window=7).mean() ,summer2021_title + " 7 Day Rolling Average", show=True, y_scale="symlog")


# In[ ]:


fig.savefig(f'{path}summer2021_bifrost_daily_rolling.png', bbox_inches = "tight")


# # Impact Plots

# In[ ]:


fig, ax, dct = vs.plot_major_restatement_impacts(df, data_title, show=False, percent=True)


# In[ ]:


fig.savefig(f'{path}impact.png', bbox_inches = "tight")


# In[ ]:


fig, ax, dct = vs.plot_major_restatement_impacts(df.diff(), data_title + " (Daily)", show=False,figsize=(14,20), small_multiples=True, percent=True, y_scale="symlog")


# In[ ]:


fig.savefig(f'{path}impact_daily.png', bbox_inches = "tight")


# In[ ]:


fig, ax, dct = vs.plot_major_restatement_impacts(
    df.diff().rolling(window=7).mean(),
    data_title + " 7 Day Average", show=False,figsize=(14,20), small_multiples=True, percent=True, y_scale="symlog")


# In[ ]:


fig.savefig(f'{path}impact_daily_rolling.png', bbox_inches = "tight")


# ## Final Versus Original

# In[ ]:


fig, ax, dct = vs.plot_final_to_original(df, data_title, show=False, net=True, percent=True) #, y_scale="symlog")
fig.savefig(f'{path}fvo.png', bbox_inches = "tight")


# In[ ]:


fig, ax, dct = vs.plot_final_to_original(
    df.diff(), data_title + " (Daily)", show=False, net=True, percent=True, y_scale="symlog")
fig.savefig(f'{path}fvo_daily.png', bbox_inches = "tight")


