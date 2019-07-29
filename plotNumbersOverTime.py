import numpy as np
import pandas as pd
import gensim
import sys
import os
from gensim.test.utils import datapath
from gensim import corpora
from sqlalchemy import create_engine
from tokenizer import *
import warnings
import json
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc = {'figure.figsize':(16,9)})
#can change default below (larger lines and labels)
#sns.set_context('talk')

#import argument parser
import argparse
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--type", help="Specify whether to use model trained with posts or comments. Default is comments.", choices=['posts','comments'], default='comments')
arg_parser.add_argument("--pages", help="Group of pages to use. Default is guy.", choices=['all','nh','guy']) 
arg_parser.add_argument("--start_date", help="Start date for trained model in format YYYY-MM-DD. Default is 2019-04-01.", default="2019-04-01")
arg_parser.add_argument("--end_date", help="End date for trained model in format YYYY-MM-DD. Default is 2019-06-22.", default="2019-06-22")
arg_parser.add_argument("--ignore", help="Ignore warnings", action="store_true")
args = arg_parser.parse_args()

#read off input values
if args.ignore:
    print('Ignoring all warnings...')
    warnings.filterwarnings("ignore", category=DeprecationWarning)
print("Using %s for analysis" % args.type)

#pages used for facebook pull (USEmbassy included in all intentionally)
if args.pages == 'all':
    page_ids = ['AFSOUTHNewHorizons','USEmbassyGeorgetown','southcom','dpiguyana','AFSouthern','NewsSourceGuyana','655452691211411','kaieteurnewsonline','demwaves','CapitolNewsGY','PrimeNewsGuyana','INews.Guyana','stabroeknews','NCNGuyanaNews','dailynewsguyana','actionnewsguyana','gychronicle','gytimes','newsroomgy']
elif args.pages == 'nh':
    page_ids = ['AFSOUTHNewHorizons','USEmbassyGeorgetown','southcom','AFSouthern']
elif args.pages == 'guy':
    page_ids = ['USEmbassyGeorgetown','dpiguyana','NewsSourceGuyana','655452691211411','kaieteurnewsonline','demwaves','CapitolNewsGY','PrimeNewsGuyana','INews.Guyana','stabroeknews','NCNGuyanaNews','dailynewsguyana','actionnewsguyana','gychronicle','gytimes','newsroomgy'] 

engine=create_engine('sqlite:///./nh19_fb.db')
df = pd.read_sql("""select * from {0} where created_time >= '{1}' and created_time <= '{2}'""".format(args.type, args.start_date, args.end_date), engine)

relevant_df = df[df['page'].isin(page_ids)]

#count per period overall 
time_count_df = relevant_df
time_count_df['created_time'] = pd.to_datetime(time_count_df.created_time)
time_count_df = time_count_df.groupby([pd.Grouper(key='created_time', freq='M')]).size().reset_index(name='counts')
time_count_df = time_count_df.fillna(0)
time_count_df['created_time'] = time_count_df['created_time'].dt.strftime("%Y-%m-%d")
print('count {0} over time dataframe'.format(args.type))
print(time_count_df.head())
plt.figure()
ax = sns.pointplot(x='created_time',y='counts', ci=None, data=time_count_df)
ax.grid(True)
ax.tick_params(axis = 'both', which = 'major', labelsize = 14)
plt.title('Number of {0} Over Time for {1} pages'.format(args.type.capitalize(),args.pages.upper()), fontsize=24)
plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
plt.subplots_adjust(bottom=0.2)
plt.xlabel("Created Date", fontsize = 18)
plt.ylabel("Count", fontsize = 18)

#count by page overall
page_stats = relevant_df.groupby("page")
plt.figure()
page_stats.size().sort_values(ascending=False).plot.bar()
plt.xticks(rotation=40, ha='right', rotation_mode='anchor')
plt.subplots_adjust(bottom=0.2)
plt.xlabel("Facebook Page", fontsize = 20)
plt.ylabel("Number of {0}".format(args.type.capitalize()), fontsize = 20)
ax = plt.gca()
ax.tick_params(axis = 'both', which = 'major', labelsize = 16)
plt.title("Number of {0} Per Facebook Page".format(args.type.capitalize()), fontsize = 24)

#count per period for each page 
time_count_page_df = relevant_df
time_count_page_df['created_time'] = pd.to_datetime(time_count_page_df.created_time)
time_count_page_df = time_count_page_df.groupby([pd.Grouper(key='created_time', freq='M'),'page']).size().reset_index(name='counts')
time_count_page_df = time_count_page_df.fillna(0)
time_count_page_df['created_time'] = time_count_page_df['created_time'].dt.strftime("%Y-%m-%d")
print('count {0} per page over time dataframe'.format(args.type))
print(time_count_page_df.head())
plt.figure()
ax = sns.pointplot(x='created_time',y='counts', hue='page', ci=None, data=time_count_page_df, palette=sns.color_palette("hls",len(page_ids)+1))
ax.grid(True)
ax.tick_params(axis = 'both', which = 'major', labelsize = 14)
plt.title('Number of {0} Over Time for {1} pages'.format(args.type.capitalize(),args.pages.upper()), fontsize=24)
plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
plt.subplots_adjust(bottom=0.2)
plt.xlabel("Created Date", fontsize = 18)
plt.ylabel("Count", fontsize = 18)

plt.show()
